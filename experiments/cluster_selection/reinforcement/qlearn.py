"""Construct clusters with terminal pruning rewards and replayed Q backups.

Each episode grows one cluster, then contracts it under fixed current domains.
Intermediate additions have zero reward. Gamma is one; affordability is bounded
by the existing construction masks and contraction evaluator, not a size penalty.
"""
import argparse
from collections import defaultdict, deque
from copy import deepcopy
from hashlib import sha256
import json
from math import ceil, isfinite
from pathlib import Path
from random import Random
from time import perf_counter
import torch
from torch.nn.functional import smooth_l1_loss
from ..candidate_evaluation_tnreason import SummaryClusterCandidate
from ..candidate_generation import ClusterCandidateGenerator
from ..runtime import apply_candidate
from .experiment import evaluate_roots, instance_hashes, instances, solve
from .learn import _signature, _teacher_search_steps, causal_candidates, measure, mutations, training_states
from .model import Policy
from .train import _closed, _save


def _states(trace):
    """Construction states before each action; the final action must terminate."""
    keys, opened, phase, states, done = [], [], False, [], False
    for kind, idx in trace:
        if done:
            raise ValueError("Actions after construction termination")
        states.append((tuple(keys), tuple(opened), phase))
        if kind == "core":
            keys.append(idx)
        elif kind == "open":
            opened.append(idx)
        elif kind == "finish" and not phase:
            phase = True
        elif kind in {"finish", "skip"}:
            done = True
        else:
            raise ValueError("Unknown construction action")
    if not done:
        raise ValueError("Replay requires a completed construction")
    return states


class Replay:
    """Completed episodes, including no-ops; sample puzzles before episodes."""

    def __init__(self, capacity=2048, n_step=8):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (capacity, n_step)):
            raise ValueError("Replay capacity and backup length must be positive integers")
        self.capacity, self.n_step = capacity, n_step
        self.episodes = deque(maxlen=capacity)

    def add(self, puzzle, domains, trace, reward, **outcome):
        if not isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("Expected normalized pruning reward in [0, 1]")
        if isinstance(domains, dict):
            domains = domains.values()
        snapshot = tuple(tuple(values) for values in domains)
        trace = tuple(tuple(action) for action in trace)
        _states(trace)
        self.episodes.append(dict(outcome, puzzle=puzzle, domains=snapshot, trace=trace, reward=float(reward)))

    def sample(self, rng, count):
        groups = defaultdict(list)
        for episode in self.episodes:
            groups[episode["puzzle"]].append(episode)
        if not groups:
            raise ValueError("Cannot sample empty replay")
        puzzles, result = list(groups), []
        for _ in range(count):
            pool = groups[rng.choice(puzzles)]
            positive = [item for item in pool if item["reward"] > 0]
            episode = rng.choice(positive if positive and rng.random() < .5 else pool)
            offset = len(episode["trace"]) - 1 if rng.random() < .25 else rng.randrange(len(episode["trace"]))
            result.append((episode, offset))
        return result


def q_loss(model, target, replay, batch_size, rng, n_step=8):
    """Masked Q backups with achievable-return bounds from successful episodes.

    A completed no-op labels its finish action zero. Earlier actions bootstrap
    one step, retaining alternative continuations. Successful paths supply an
    achievable lower bound and up to n-step backups. Targets are frozen and
    bounded by the known [0,1] terminal-reward range; no length discount applies.
    """
    encoded, target_encoded, losses = {}, {}, []
    for episode, offset in replay.sample(rng, batch_size):
        domains, trace = episode["domains"], episode["trace"]
        if domains not in encoded:
            encoded[domains] = model.encode(domains)
        sizes, states = [len(values) for values in domains], _states(trace)
        keys, opened, phase = states[offset]
        actions, values = model.q_values(encoded[domains], sizes, list(keys), list(opened), phase)
        prediction = values[actions.index(tuple(trace[offset]))]
        with torch.no_grad():
            if offset == len(trace) - 1:
                utility = prediction.new_tensor(episode["reward"])
            else:
                if domains not in target_encoded:
                    target_encoded[domains] = target.encode(domains)
                keys, opened, phase = states[offset + 1]
                _, future = target.q_values(target_encoded[domains], sizes, list(keys), list(opened), phase)
                utility = future.max().clamp(0., 1.)
                if episode["reward"] > 0:
                    # A demonstrated completion is a lower bound, including near
                    # its end; preserve higher-valued alternative continuations.
                    utility = utility.clamp_min(episode["reward"])
                    following = offset + n_step
                    if following < len(trace) and n_step > 1:
                        keys, opened, phase = states[following]
                        _, future = target.q_values(target_encoded[domains], sizes, list(keys), list(opened), phase)
                        utility = torch.maximum(utility, future.max().clamp(0., 1.))
        losses.append(smooth_l1_loss(prediction, utility))
    return torch.stack(losses).mean()


@torch.no_grad()
def construct(model, domains, rng, epsilon):
    """Epsilon-greedy free construction; no teachers or forced prefixes."""
    encoded = model.encode(domains)
    sizes = [len(domains[color]) for color in model.network["colors"]]
    keys, opened, phase, trace = [], [], False, []
    while True:
        actions, values = model.q_values(encoded, sizes, keys, opened, phase)
        choice = rng.randrange(len(actions)) if rng.random() < epsilon else int(values.argmax())
        action = actions[choice]
        trace.append(action)
        kind, idx = action
        if kind == "core":
            keys.append(idx)
        elif kind == "open":
            opened.append(idx)
        elif kind == "finish" and not phase:
            phase = True
        else:
            candidate = (SummaryClusterCandidate("q", tuple(model.keys[i] for i in sorted(keys)),
                         tuple(model.network["colors"][i] for i in sorted(opened))) if kind == "finish" else None)
            return candidate, trace


def fit(data, *, output, epochs=8, width=32, rounds=2, lr=.0003, seed=0, puzzles_per_epoch=32,
        rollouts=4, steps=3, teacher_candidates=32, teacher_mutations=12, teacher_probes=0,
        batch_size=32, updates_per_state=16, n_step=8, target_update=100, replay_capacity=8192,
        epsilon_start=.3, epsilon_end=.05, validation_seconds=300., search_nodes=10000,
        max_total_work=None, cluster_attempts=512, initial_checkpoint=None,
        train_indices=None, validation_indices=None, device="cpu",
        validation_interval=1, checkpoint_every=0, progress_every=100, cycle_puzzles=False,
        teacher_max_steps=None, nested_teacher=False, **limits):
    """Seed replay from saved/measured teachers, then alternate exploration and Q fitting."""
    output = Path(output)
    paths = [output, output.with_stem(output.stem + ".best"), output.with_stem(output.stem + ".initial")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Output already exists; choose a fresh experiment path")
    teacher_max_steps = _teacher_search_steps(teacher_candidates, teacher_max_steps, nested_teacher)
    counts = (epochs, width, rounds, rollouts, steps, batch_size,
              updates_per_state, n_step, target_update, replay_capacity, search_nodes, cluster_attempts,
              validation_interval, progress_every)
    if (any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in counts)
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0
                   for v in (teacher_candidates, teacher_mutations, teacher_probes, checkpoint_every, puzzles_per_epoch))
            or not isfinite(lr) or lr <= 0 or not isfinite(validation_seconds) or validation_seconds <= 0
            or not isinstance(cycle_puzzles, bool)
            or not 0 <= epsilon_end <= epsilon_start <= 1
            or max_total_work is not None and (isinstance(max_total_work, bool)
                or not isinstance(max_total_work, int) or max_total_work < 1)):
        raise ValueError("Invalid Q-learning parameters")
    training_device = torch.device(device)
    if training_device.type not in {"cpu", "cuda"}:
        raise ValueError("Q-learning supports CPU or CUDA training")
    if training_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable: use CUDA-enabled PyTorch on an allocated GPU")
    snapshots = {epoch: output.with_stem(output.stem + f".epoch{epoch:04d}")
                 for epoch in range(checkpoint_every, epochs + 1, checkpoint_every)} if checkpoint_every else {}
    if any(path.exists() for path in snapshots.values()):
        raise FileExistsError("Epoch snapshot already exists; choose a fresh experiment path")
    base, state, pools, validation = training_states(data, train_indices, validation_indices, progress_every)
    if not validation:
        raise ValueError("Q-learning requires validation instances for DFS checkpoint selection")
    torch.manual_seed(seed)
    rng = Random(seed)
    config = dict(max_cores=32, max_open_colors=6, max_work=1_000_000, max_size=100_000,
        compact_domains=True, contextual=True, target_first=True, cyclic_only=True, min_active=2,
        heuristic_candidates=0, exploration=0., q_learning=True, cluster_attempts=cluster_attempts,
        max_total_work=max_total_work)
    config.update(limits)
    if not config["contextual"] or not config["q_learning"]:
        raise ValueError("Q-learning requires contextual raw action values")
    model = Policy(base["network"], width, rounds, **config)
    model.reward_colors, model.beta = base["reward_colors"], 0.
    initial_domains, _ = instances(data)
    exposure = instance_hashes(data, [initial_domains[i] for i in pools])
    parent, saved = None, None
    data_hash = sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if initial_checkpoint is not None:
        initial_checkpoint = Path(initial_checkpoint)
        saved = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
        inherited = saved.get("training", {}).get("train_instance_hashes")
        if (saved.get("network") != model.network or inherited is None
                or set(inherited) & instance_hashes(data, validation)):
            raise ValueError("Initialization must match the network and exclude validation exposure")
        model.load_state_dict(saved["weights"])
        exposure.update(inherited)
        if not saved.get("config", {}).get("q_learning"):
            torch.nn.init.normal_(model.head[-1].weight, std=.001)
            torch.nn.init.zeros_(model.head[-1].bias)
        parent = dict(path=str(initial_checkpoint), sha256=sha256(initial_checkpoint.read_bytes()).hexdigest())
    model.to(training_device)
    output.parent.mkdir(parents=True, exist_ok=True)
    target = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    replay = Replay(replay_capacity, n_step)
    reused = 0
    if (saved is not None and saved.get("config", {}).get("q_learning")
            and saved.get("training", {}).get("data_sha256") == data_hash
            and saved.get("training", {}).get("reward_units") == "entropy_fraction"
            and saved.get("training", {}).get("objective") == "pruning"
            and saved.get("replay", {}).get("version") == 1
            and saved["config"] == model.config and saved.get("reward_colors") == model.reward_colors):
        for episode in saved.get("replay", {}).get("episodes", ()):
            replay.add(**episode)
            reused += 1
    measured, generated, history, updates = {}, set(), [], 0
    for episode in replay.episodes:
        signature = (tuple(sorted(episode["cores"])), tuple(sorted(episode["outputs"]))) if episode["cores"] else None
        measured[episode["domains"], signature] = dict(reward=episode["reward"],
            useful=int(episode["reward"] > 0), gain=episode["gain"], contradictions=episode["contradiction"],
            rejected=episode["rejected"], work=episode["work"],
            domains=dict(zip(model.network["colors"], episode["domains_after"])))
    counters = defaultdict(float)
    generator = ClusterCandidateGenerator(state.constraint_network, max_candidates=max(1, teacher_candidates),
        max_steps=teacher_max_steps,
        **{key: config[key] for key in ("max_cores", "max_open_colors", "max_work", "max_size")})
    generate = generator.generate_nested if nested_teacher else generator.generate

    def record(puzzle, candidate, trace, source, known_satisfiable=False):
        snapshot = tuple(state.domains[color] for color in model.network["colors"])
        signature = None if candidate is None else _signature(candidate)
        key = snapshot, signature
        if key not in measured:
            metrics = measure(model, state, candidate, objective="pruning", record_domains=True,
                              known_satisfiable=known_satisfiable)
            # Alternate legal construction orders reuse the same measured result.
            if len(measured) >= max(4096, replay_capacity * 2):
                measured.pop(next(iter(measured)))
            measured[key] = metrics
            counters["measurements"] += int(candidate is not None)
            counters["measured_useful"] += metrics["useful"]
            counters["measured_noop"] += int(candidate is not None and not metrics["useful"] and not metrics["rejected"])
            counters["measured_rejected"] += metrics["rejected"]
        metrics = measured[key]
        replay.add(puzzle, snapshot, trace, metrics["reward"], source=source,
            cores=() if candidate is None else candidate.core_keys,
            outputs=() if candidate is None else candidate.open_colors,
            domains_after=tuple(metrics["domains"][color] for color in model.network["colors"]),
            gain=metrics["gain"], contradiction=metrics["contradictions"], rejected=metrics["rejected"],
            work=metrics["work"])
        counters[source + "_episodes"] += 1
        return metrics

    def teachers(puzzle, proposals, known_satisfiable=False, source="teacher"):
        scored = []
        with torch.no_grad():
            for candidate in proposals:
                try:
                    trace = model.candidate_trace(candidate, state.domains, rng=rng)
                except ValueError:
                    counters["illegal_teachers"] += 1
                    continue
                scored.append((candidate, record(puzzle, candidate, trace, source, known_satisfiable)))
        return scored

    def train_updates(count):
        nonlocal updates
        losses = []
        model.train()
        for _ in range(count):
            if not replay.episodes:
                break
            optimizer.zero_grad()
            loss = q_loss(model, target, replay, batch_size, rng, n_step)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach()))
            if updates % target_update == 0:
                target.load_state_dict(model.state_dict())
        return sum(losses) / max(1, len(losses))

    def selection():
        model.eval()
        # Deployment loads checkpoints on CPU; validate those same numerical
        # decisions and keep the sequential MAC controller off the training GPU.
        validator = model if training_device.type == "cpu" else deepcopy(model).cpu().eval()
        result = evaluate_roots(state.constraint_network, validation, validator, seconds=validation_seconds,
                                total_work=max_total_work, cluster_attempts=cluster_attempts)
        compact = {key: value for key, value in result.items() if key != "rows"}
        if compact["contradictions"] and "puzzles" in data:
            raise AssertionError("Validation contraction contradicted a known satisfiable puzzle")
        greedy = []
        for domains in validation:
            probe = state.constraint_network.create_mac_state(domains)
            if probe.propagate() and not probe.is_solved():
                greedy.append(measure(validator, probe, validator.propose(probe.domains), objective="pruning"))
        compact.update(greedy_trials=len(greedy), greedy_useful=sum(row["useful"] for row in greedy),
                       greedy_reward=sum(row["reward"] for row in greedy) / max(1, len(greedy)))
        rows = [solve(state.constraint_network, domains, "trained", validator, nodes=search_nodes,
                      seconds=validation_seconds, total_work=max_total_work, cluster_attempts=cluster_attempts)
                for domains in validation]
        if any(row["status"] == "invalid" or "puzzles" in data and row["status"] == "unsat" for row in rows):
            raise AssertionError("Invalid validation search outcome")
        complete = sum(row["status"] in {"solved", "unsat"} for row in rows)
        nodes = sum(row["nodes"] or 0 for row in rows) if complete == len(rows) else None
        compact.update(search_complete=complete, search_nodes=nodes,
                       search_statuses=[row["status"] for row in rows],
                       search_nodes_by_instance=[row["nodes"] for row in rows])
        return compact, (complete, -nodes if nodes is not None else -float("inf"))

    epoch_puzzles = min(puzzles_per_epoch or len(pools), len(pools))
    chunks = ceil(len(pools) / epoch_puzzles)
    epoch_unit = "chunk" if cycle_puzzles and chunks > 1 else "epoch"
    schedule = ("full pass" if chunks == 1 else f"{chunks} chunks per full pass" if cycle_puzzles else "sampled pass")
    planned_visits = ((epochs // chunks) * len(pools) + min(epochs % chunks * epoch_puzzles, len(pools))
                      if cycle_puzzles else epochs * epoch_puzzles)
    progress = dict(puzzle_visits=0, state_exposures=0, puzzle_passes=0., unique_puzzles_visited=0)
    visited_puzzles = set()
    settings = dict(method="construction_q_learning", objective="pruning", reward_units="entropy_fraction",
        gamma=1., seed=seed, train_puzzles=sorted(pools), train_instance_hashes=sorted(exposure), parent=parent,
        device=str(training_device), validation_device="cpu", epochs=epochs, planned_epochs=epochs,
        validation_interval=validation_interval, checkpoint_every=checkpoint_every,
        puzzles_per_epoch=puzzles_per_epoch, puzzles_per_epoch_effective=epoch_puzzles,
        dataset_instances=len(initial_domains), validation_instances=len(validation),
        train_puzzle_count=len(pools), train_context_count_initial=sum(map(len, pools.values())),
        planned_puzzle_visits=planned_visits, planned_puzzle_passes=planned_visits / len(pools),
        cycle_puzzles=cycle_puzzles, epoch_unit=epoch_unit, chunks_per_pass=chunks if cycle_puzzles else None,
        progress_every=progress_every,
        data_sha256=data_hash, replay_reused=reused,
        cluster_attempts=cluster_attempts, n_step=n_step, batch_size=batch_size, target_update=target_update,
        replay_capacity=replay_capacity, updates_per_state=updates_per_state, lr=lr,
        search_nodes=search_nodes, validation_seconds=validation_seconds,
        epsilon_start=epsilon_start, epsilon_end=epsilon_end, rollouts=rollouts, steps=steps,
        teacher_candidates=teacher_candidates, teacher_mutations=teacher_mutations, teacher_probes=teacher_probes,
        teacher_max_steps=teacher_max_steps, nested_teacher=nested_teacher,
        backup="masked_one_step_and_reachable_lower_bounds")

    current_epoch = last_validation_epoch = 0

    def save(path, result):
        _save(path, model, history, 0., 1., result, cost_metric="none", training=dict(settings, **progress, updates=updates,
              train_context_count=sum(map(len, pools.values())),
              epoch=current_epoch, last_validation_epoch=last_validation_epoch),
              replay=dict(version=1, episodes=list(replay.episodes)),
              optimizer=optimizer.state_dict(), target_weights=target.state_dict(), measurements=dict(counters))

    print(f"Q-learning on {training_device}: terminal pruning, gamma=1, unforced construction", flush=True)
    print(f"training data: {len(initial_domains)} dataset instances including {len(validation)} validation; "
          f"{len(pools)} unresolved training puzzles, {sum(map(len, pools.values()))} contexts; "
          f"up to {epoch_puzzles} puzzles per {epoch_unit} ({schedule})", flush=True)
    if training_device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(training_device)}; PyTorch {torch.__version__}; "
              f"CUDA {torch.version.cuda}; validation on CPU", flush=True)
    print("validating initialization", flush=True)
    initial, best = selection()
    print(f"initial validation: greedy useful {initial['greedy_useful']}/{initial['greedy_trials']}, "
          f"completed DFS {initial['search_complete']}/{len(validation)}, nodes {initial['search_nodes']}", flush=True)
    for path in paths[1:]:
        save(path, initial)
    # Existing measurements are proposals, not trusted labels under a new grader.
    for sample in data.get("samples", ()):
        if sample.get("puzzle") not in pools or not sample.get("candidates"):
            continue
        with _closed(state, sample, model.network["colors"]):
            proposals = [SummaryClusterCandidate("saved", tuple(model.keys[i] for i in row["cores"]),
                tuple(model.network["colors"][i] for i in row["open"])) for row in sample["candidates"]]
            teachers(sample["puzzle"], proposals, source="saved")
    print(f"seed replay: {len(replay.episodes)} episodes, "
          f"{sum(episode['reward'] > 0 for episode in replay.episodes)} useful", flush=True)
    print(f"structural teacher: {'nested' if nested_teacher else 'direct'}, up to {teacher_candidates} proposals; "
          f"largest-stage search budget {teacher_max_steps} steps", flush=True)
    pending_puzzles = []
    for epoch in range(epochs):
        began = last_progress = perf_counter()
        before, before_progress, before_updates = dict(counters), dict(progress), updates
        trials = useful = 0
        total_reward, losses = 0., []
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * epoch / max(1, epochs - 1)
        if not cycle_puzzles or not pending_puzzles:
            pending_puzzles = list(pools)
            rng.shuffle(pending_puzzles)
        puzzle_order = pending_puzzles[:epoch_puzzles]
        pending_puzzles = pending_puzzles[epoch_puzzles:] if cycle_puzzles else []
        print(f"{epoch_unit} {epoch + 1}/{epochs}: starting {len(puzzle_order)} puzzles, epsilon {epsilon:.3f}", flush=True)
        for puzzle_number, puzzle in enumerate(puzzle_order, 1):
            progress["puzzle_visits"] += 1
            visited_puzzles.add(puzzle)
            roots = [item for item in pools[puzzle] if item["root"]]
            sample = roots[0] if roots and rng.random() < .5 else rng.choice(pools[puzzle])
            with _closed(state, sample, model.network["colors"]):
                for _ in range(steps):
                    progress["state_exposures"] += 1
                    snapshot = tuple(state.domains.values())
                    scored = []
                    if snapshot not in generated:
                        proposals = list(generate(state.domains, max_candidates=teacher_candidates,
                            cyclic_only=True, compact_domains=config["compact_domains"]))
                        structural = {_signature(candidate) for candidate in proposals}
                        counters["teacher_structural_calls"] += 1
                        counters["teacher_structural_candidates"] += len(proposals)
                        proposals += causal_candidates(state, model.reward_colors, rng, teacher_probes,
                                                       max_cores=config["max_cores"])
                        scored = teachers(puzzle, proposals, sample["known_satisfiable"])
                        counters["teacher_structural_useful"] += len({_signature(candidate) for candidate, metrics in scored
                            if metrics["useful"] and _signature(candidate) in structural})
                        seeds = [candidate for candidate, metrics in scored if metrics["useful"]] or proposals
                        if seeds:
                            scored += teachers(puzzle, mutations(model, state, seeds, rng, teacher_mutations),
                                               sample["known_satisfiable"])
                        generated.add(snapshot)
                    # Teachers seed replay; exploration always constructs from the
                    # empty cluster. Its success rate has no changing forced prefix.
                    losses.append(train_updates(updates_per_state))
                    for _ in range(rollouts):
                        candidate, trace = construct(model, state.domains, rng, epsilon)
                        metrics = record(puzzle, candidate, trace, "policy", sample["known_satisfiable"])
                        trials += 1
                        useful += metrics["useful"]
                        total_reward += metrics["reward"]
                        scored.append((candidate, metrics))
                    losses.append(train_updates(updates_per_state))
                    positive = [(c, m) for c, m in scored if c is not None and m["useful"]]
                    if not positive:
                        break
                    candidate, _ = max(positive, key=lambda item: item[1]["reward"])
                    result = apply_candidate(state, candidate, compact_domains=config["compact_domains"],
                                             max_work=config["max_work"], max_size=config["max_size"])
                    if result.contradiction or state.is_solved() or not result.removed_values:
                        break
                    snapshot = tuple(state.domains.values())
                    if len(pools[puzzle]) < 12 and all(item["domains"] != snapshot for item in pools[puzzle]):
                        pools[puzzle].append(dict(domains=snapshot, root=False,
                                                 known_satisfiable=sample["known_satisfiable"]))
            if puzzle_number % progress_every == 0 or perf_counter() - last_progress >= 60:
                print(f"{epoch_unit} {epoch + 1}/{epochs}: puzzles {puzzle_number}/{len(puzzle_order)}, "
                      f"states {progress['state_exposures'] - before_progress['state_exposures']}, "
                      f"updates {updates - before_updates} ({updates} total), unforced useful {useful}/{trials}, "
                      f"Q loss {sum(losses) / max(1, len(losses)):.4f}, "
                      f"passes {progress['puzzle_visits'] / len(pools):.2f}", flush=True)
                last_progress = perf_counter()
        progress.update(puzzle_passes=progress["puzzle_visits"] / len(pools),
                        unique_puzzles_visited=len(visited_puzzles))
        current_epoch = epoch + 1
        validated = current_epoch % validation_interval == 0 or current_epoch == epochs
        result, rank = selection() if validated else (None, None)
        if validated:
            last_validation_epoch = current_epoch
        history.append(dict(epoch=epoch + 1, seconds=perf_counter() - began, trials=trials, useful=useful,
            reward=total_reward, epsilon=epsilon, q_loss=sum(losses) / max(1, len(losses)), updates=updates,
            epoch_updates=updates - before_updates, puzzle_visits=progress["puzzle_visits"] - before_progress["puzzle_visits"],
            total_puzzle_visits=progress["puzzle_visits"], puzzle_passes=progress["puzzle_passes"],
            state_exposures=progress["state_exposures"] - before_progress["state_exposures"],
            total_state_exposures=progress["state_exposures"], train_context_count=sum(map(len, pools.values())),
            replay_episodes=len(replay.episodes), measurements=counters["measurements"],
            **{key: counters[key] - before.get(key, 0) for key in
               ("teacher_structural_calls", "teacher_structural_candidates", "teacher_structural_useful")},
            epoch_measurements=counters["measurements"] - before.get("measurements", 0), validation=result))
        if validated and rank > best:
            best = rank
            save(paths[1], result)
        selection_record = result if validated else dict(source="skipped", epoch=current_epoch,
                                                         last_validation_epoch=last_validation_epoch)
        save(output, selection_record)
        if current_epoch in snapshots:
            path = snapshots[current_epoch]
            if path.exists():
                raise FileExistsError(f"Epoch snapshot already exists: {path}")
            save(path, selection_record)
        validation_text = (f"greedy validation {result['greedy_useful']}/{result['greedy_trials']}, "
                           f"DFS nodes {result['search_nodes']}" if validated else "validation skipped")
        print(f"{epoch_unit} {epoch + 1}/{epochs}: unforced useful {useful}/{trials}, "
              f"{validation_text}, replay {len(replay.episodes)}, "
              f"Q loss {history[-1]['q_loss']:.4f}, puzzles {len(puzzle_order)}/{len(pools)}, "
              f"states {history[-1]['state_exposures']}, updates {history[-1]['epoch_updates']} "
              f"({updates} total), passes {progress['puzzle_passes']:.2f}, "
              f"structural proposals {int(history[-1]['teacher_structural_candidates'])} on "
              f"{int(history[-1]['teacher_structural_calls'])} fresh states "
              f"({int(history[-1]['teacher_structural_useful'])} useful)", flush=True)
    return model.eval(), history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    for key, default in dict(epochs=8, width=32, rounds=2, lr=.0003, seed=0, puzzles_per_epoch=32,
        rollouts=4, steps=3, teacher_candidates=32, teacher_mutations=12, teacher_probes=0,
        batch_size=32, updates_per_state=16, n_step=8, target_update=100, replay_capacity=8192,
        epsilon_start=.3, epsilon_end=.05, validation_seconds=300., search_nodes=10000,
        cluster_attempts=512, max_cores=32, max_open_colors=6, max_work=1_000_000, max_size=100_000,
        validation_interval=1, checkpoint_every=0, progress_every=100).items():
        help_text = {"puzzles_per_epoch": "Puzzles per epoch; 0 visits every unresolved training puzzle once",
                     "progress_every": "Print training progress after this many puzzles (also every 60 seconds)"}
        parser.add_argument("--" + key.replace("_", "-"), type=type(default), default=default,
                            help=help_text.get(key))
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--nested-teacher", action="store_true",
                        help="Keep structural proposal prefixes while expanding doubling stages from 32 candidates")
    parser.add_argument("--teacher-max-steps", type=int,
                        help="Largest structural search-stage budget; default 8192, or 256 per rounded-up candidate in nested mode")
    parser.add_argument("--device", default="cpu", help="cpu or cuda; CUDA uses HTCondor's visible GPU")
    parser.add_argument("--train-indices", type=int, nargs="+")
    parser.add_argument("--validation-indices", type=int, nargs="+")
    parser.add_argument("--cycle-puzzles", action="store_true",
                        help="Consume a shuffled dataset in epoch-sized chunks before repeating any puzzle")
    parser.add_argument("--value-encoding", action="store_true")
    parser.add_argument("--relational-context", action="store_true")
    parser.add_argument("--relational-rounds", type=int, default=6)
    args = vars(parser.parse_args())
    path = args.pop("data")
    torch.set_num_threads(1)
    fit(json.loads(path.read_text()), **args)


if __name__ == "__main__":
    main()
