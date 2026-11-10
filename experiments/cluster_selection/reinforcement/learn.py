"""Learn useful constructions from measured teachers and fresh one-step trials.

Useful contractions supply the next training state. The DFS-node objective uses
completed counterfactual searches; unchanged domains earn zero, never delayed credit.
"""
import argparse
from collections import defaultdict, deque
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
import json
from math import ceil, isfinite, log, sqrt
from pathlib import Path
from random import Random
from time import perf_counter
import torch
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from ..candidate_evaluation_mac import evaluate_candidate, prepare_candidate
from ..candidate_evaluation_tnreason import SummaryClusterCandidate
from ..candidate_generation import ClusterCandidateGenerator
from ..runtime import apply_candidate
from .experiment import evaluate_roots, fingerprints, instance_hashes as _instance_hashes, instances, solve
from .model import Policy
from .train import _closed, _save, _state


def _signature(candidate):
    return tuple(sorted(candidate.core_keys)), tuple(sorted(candidate.open_colors))


def _teacher_search_steps(candidates, maximum, nested):
    if (not isinstance(candidates, int) or isinstance(candidates, bool) or candidates < 0
            or not isinstance(nested, bool)
            or maximum is not None and (not isinstance(maximum, int)
                or isinstance(maximum, bool) or maximum < 1)):
        raise ValueError("Expected nonnegative teacher count, positive search steps and boolean nested mode")
    if maximum is not None:
        return maximum
    stage = 32
    while stage < candidates:
        stage *= 2
    return 256 * stage if nested else 8192


def causal_candidates(state, colors, rng, limit, *, max_cores):
    """Explain failed-value GAC probes; callers must still measure contractions.

    A reduction depends conservatively on its source core and all previous
    reductions in that core's scope. Their union at a conflict is a sufficient
    witness under the original domains. Only a separate scratch state is traced.
    """
    if limit < 0 or max_cores < 1:
        raise ValueError("Expected nonnegative probes and positive core limit")
    if not limit:
        return []
    root = state.snapshot_domains()
    assumptions = [(color, value) for color in colors if len(root[color]) > 1
                   for value in root[color]]
    rng.shuffle(assumptions)
    probe = ConstraintNetwork(state._cores, maintain_indices=True).create_mac_state(root)
    probe.propagate()
    if probe.contradiction:
        return []
    if probe.snapshot_domains() != root:
        raise ValueError("Causal teachers require GAC-closed input domains")
    original = probe._set_domain
    candidates, seen = [], set()

    def traced(color, revised, source=None):
        nonlocal conflict
        if revised != probe.domains[color]:
            if source is not None:
                causes = {source}
                for other in probe._cores[source].colors:
                    causes.update(reasons[other])
                reasons[color].update(causes)
            if not revised:
                conflict = reasons[color].copy()
        return original(color, revised, source)

    probe._set_domain = traced
    try:
        for color, value in assumptions[:limit]:
            reasons, conflict = {key: set() for key in root}, None
            probe.push_level()
            try:
                probe.assign(color, value)
                probe.propagate()
                if conflict and len(conflict) <= max_cores:
                    candidate = SummaryClusterCandidate("causal_" + color, tuple(sorted(conflict)), (color,))
                    signature = _signature(candidate)
                    if signature not in seen:
                        candidates.append(candidate)
                        seen.add(signature)
            finally:
                probe.pop_level()
    finally:
        del probe.__dict__["_set_domain"]
    return candidates


def training_states(data, train_indices=None, validation_indices=None, progress_every=0):
    """Close roots once, balance puzzles, and exclude validation puzzle states."""
    if not isinstance(progress_every, int) or isinstance(progress_every, bool) or progress_every < 0:
        raise ValueError("Preparation progress interval must be a nonnegative integer")
    raw = data["network"]
    base = {**data, "reward_colors": data.get("reward_colors", raw["colors"])}
    base.setdefault("samples", [dict(domains=[list(range(dim)) for dim in raw["dimensions"]])])
    state = _state(base)
    initial, _ = instances(data)
    if train_indices is None:
        train_indices = data.get("train_indices")
        if train_indices is None:
            boundary = max(1, 4 * len(initial) // 5)
            train_indices = list(range(boundary))
    if validation_indices is None:
        validation_indices = data.get("validation_indices", [idx for idx in range(len(initial)) if idx not in train_indices])
    if set(train_indices) & set(validation_indices) or any(
            idx < 0 or idx >= len(initial) for idx in (*train_indices, *validation_indices)):
        raise ValueError("Training and validation indices must be valid and disjoint")
    if fingerprints(data, [initial[idx] for idx in train_indices]) & fingerprints(
            data, [initial[idx] for idx in validation_indices]):
        raise ValueError("Training and validation initial domains overlap")
    pools, seen = defaultdict(list), set()

    def add(puzzle, sample):
        snapshot = tuple(tuple(values) for values in sample["domains"])
        if snapshot not in seen:
            pools[puzzle].append({**sample, "domains": snapshot})
            seen.add(snapshot)

    if progress_every:
        print(f"GAC preparation: {len(train_indices)} training roots; "
              f"{len(validation_indices)} validation instances", flush=True)
    last_progress = perf_counter()
    for root_number, puzzle in enumerate(train_indices, 1):
        state.push_level()
        try:
            for color, values in initial[puzzle].items():
                state.restrict_domain(color, values)
            if state.propagate() and not state.is_solved():
                add(puzzle, dict(domains=list(state.domains.values()), root=True,
                                 known_satisfiable="puzzles" in data))
        finally:
            state.pop_level()
        if progress_every and (root_number % progress_every == 0 or root_number == len(train_indices)
                               or perf_counter() - last_progress >= 60):
            print(f"GAC preparation {root_number}/{len(train_indices)} roots; unresolved pools {len(pools)}", flush=True)
            last_progress = perf_counter()
    for sample in data.get("samples", ()):
        if sample.get("puzzle") in train_indices and any(len(values) > 1 for values in sample["domains"]):
            add(sample["puzzle"], {**sample, "root": False, "known_satisfiable": False})
    if not pools:
        raise ValueError("No unresolved training states; use harder initial domains")
    return base, state, pools, [initial[idx] for idx in validation_indices]


def measure(model, state, candidate, *, cost_weight=.02, solve_reward=1.,
            contradiction_reward=1., proposal_s=0., known_satisfiable=False, work_budget=None,
            search=None, objective="mixed", record_domains=False):
    """Measure pruning, optionally graded by actual counterfactual DFS savings."""
    if objective not in {"mixed", "dfs_nodes", "pruning"} or objective == "dfs_nodes" and search is None:
        raise ValueError("The dfs_nodes objective requires counterfactual search grading")
    started = perf_counter()
    metrics = dict(reward=0., reward_valid=1, gain=0, information=0., work=0, useful=0,
                   selected=int(candidate is not None), rejected=0, solved=0, contradictions=0,
                   pruned_outputs=())
    pruning_reward = 0.
    if record_domains:
        metrics["domains"] = state.snapshot_domains()
    if candidate is not None:
        limits = {key: model.config[key] for key in ("max_work", "max_size")}
        if work_budget is not None:
            limits["max_work"] = min(limits["max_work"], work_budget)
        before = state.snapshot_domains()
        try:
            result = evaluate_candidate(state, candidate, reward_colors=model.reward_colors,
                compact_domains=model.config.get("compact_domains", False), **limits)
        except ValueError as error:
            if str(error) not in ("Candidate exceeds the contraction work or tensor-size budget",
                                  "NumPy contraction supports at most 52 distinct colors"):
                raise
            metrics.update(rejected=1, reward=-cost_weight)
        else:
            if record_domains:
                metrics["domains"] = result.domains
            metrics["work"] = result.estimate.work
            metrics["contradictions"] = int(result.contradiction)
            metrics["gain"] = result.removed_domain_values if not result.contradiction else 0
            metrics["pruned_outputs"] = tuple(color for color in candidate.open_colors
                if result.contradiction or len(result.domains[color]) < len(before[color]))
            if result.contradiction:
                if known_satisfiable:
                    raise AssertionError("A sound contraction contradicted a known satisfiable training state")
                metrics.update(useful=1, reward=contradiction_reward)
                pruning_reward = 1.
            else:
                targets = set(model.reward_colors or before)
                auxiliary_weight = min(1., len(targets) / max(1, len(before) - len(targets)))
                weights = {color: 1. if color in targets else auxiliary_weight for color in before}
                remaining = sum(weights[color] * log(len(values)) for color, values in before.items())
                information = sum(weights[color] * log(len(before[color]) / len(values))
                                  for color, values in result.domains.items())
                removed = sum(len(values) - len(result.domains[color]) for color, values in before.items())
                solved = removed > 0 and all(len(values) == 1 for values in result.domains.values())
                if solved:
                    assignment = {color: values[0] for color, values in result.domains.items()}
                    if not all(core[{color: assignment[color] for color in core.colors}] != 0
                               for core in state._cores.values()):
                        raise AssertionError("Invalid training solution")
                metrics.update(information=information, useful=int(removed > 0), solved=int(solved),
                               reward=information / max(1., remaining) + solve_reward * solved)
                pruning_reward = information / remaining if remaining else 0.
            metrics["reward"] -= cost_weight * metrics["work"] / model.config["max_work"]
    metrics["cost_s"] = proposal_s + perf_counter() - started
    # Account for proposal/GAC overhead as well as tensor work. Skips and no-ops
    # have no positive reward, including when all structural teachers failed.
    metrics["reward"] -= .001 + .01 * metrics["cost_s"]
    if search is not None and objective != "pruning":
        metrics.update(search_gain=0, search_log_gain=0., search_limited=0, search_s=0.,
                       search_improved=0, search_worsened=0)
        if metrics["useful"]:
            began = perf_counter()
            baseline, after = search(before), search(result.domains)
            metrics["search_s"] = perf_counter() - began
            complete = {"solved", "unsat"}
            if baseline["status"] == "invalid" or after["status"] == "invalid":
                raise AssertionError("Invalid solution during training search grading")
            if baseline["status"] in complete and after["status"] in complete:
                if baseline["status"] != after["status"]:
                    raise AssertionError("A contraction changed satisfiability during search grading")
                metrics["search_gain"] = baseline["nodes"] - after["nodes"]
                metrics["search_improved"] = int(metrics["search_gain"] > 0)
                metrics["search_worsened"] = int(metrics["search_gain"] < 0)
                metrics["search_log_gain"] = log((baseline["nodes"] + 1) / (after["nodes"] + 1))
                metrics["reward"] += metrics["search_log_gain"]
            else:
                metrics["search_limited"] = 1
    if objective == "dfs_nodes":
        # Pruning is diagnostic, not a surrogate for the requested search goal.
        # A censored search is unknown, whereas unchanged domains are known zero.
        metrics["reward_valid"] = int(not metrics["search_limited"])
        metrics["reward"] = metrics["search_gain"]
    elif objective == "pruning":
        metrics["reward"] = min(1., max(0., pruning_reward))
    return metrics


def _positive_teachers(scored, search=False, *, objective="mixed"):
    positives = [(candidate, metrics) for candidate, metrics in scored
                 if metrics["useful"] and metrics.get("reward_valid", 1) and metrics["reward"] > 0]
    if objective == "dfs_nodes":
        positives = [item for item in positives if item[1].get("search_gain", 0) > 0]
    if search:
        saving = [item for item in positives if item[1].get("search_gain", 0) > 0]
        positives = saving or positives
    return sorted(positives, key=lambda item: item[1]["reward"], reverse=True)


def mutations(model, state, seeds, rng, count):
    """Broaden useful starts, and explore core/output additions and deletions."""
    active = {idx for idx, color in enumerate(model.network["colors"]) if len(state.domains[color]) > 1}
    eligible = {idx for idx, scope in enumerate(model.scope_sets) if len(scope & active) >= 2}
    for attempt in range(count):
        base = rng.choice(seeds)
        keys = {model.key_idx[key] for key in base.core_keys}
        if attempt % 2 == 0 and eligible:
            # Keeping a useful teacher's outputs while adding constraints cannot
            # create support. Measure affordability and actual pruning below.
            start = rng.choice(sorted(eligible))
            queue, previous, found = deque([start]), {start: None}, None
            while queue:
                key = queue.popleft()
                if key in keys:
                    found = key
                    break
                neighbors = set().union(*(model.by_color[color] for color in model.scope_sets[key] & active))
                for neighbor in sorted((neighbors & eligible) - previous.keys()):
                    previous[neighbor] = key
                    queue.append(neighbor)
            if found is None:
                continue
            while found is not None:
                keys.add(found)
                found = previous[found]
            candidate = SummaryClusterCandidate("augmentation", tuple(model.keys[idx] for idx in sorted(keys)),
                                                base.open_colors)
            try:
                model.candidate_trace(candidate, state.domains)
            except ValueError:
                continue
            yield candidate
            continue
        if len(keys) > 2 and rng.random() < .25:
            keys.remove(rng.choice(sorted(keys)))
        scope = set().union(*(model.scope_sets[key] for key in keys))
        frontier = set().union(*(model.by_color[color] for color in scope & active)) - keys
        if frontier and len(keys) < model.config["max_cores"] and rng.random() < .75:
            keys.add(rng.choice(sorted(frontier)))
        scope = set().union(*(model.scope_sets[key] for key in keys))
        colors = [model.network["colors"][idx] for idx in sorted(scope)
                  if len(state.domains[model.network["colors"][idx]]) > 1]
        rng.shuffle(colors)
        opened, size = [], 1
        for color in colors[:rng.randint(1, model.config["max_open_colors"])]:
            dim = model.network["dimensions"][model.color_idx[color]]
            if size * dim <= model.config["max_size"]:
                opened.append(color)
                size *= dim
        candidate = SummaryClusterCandidate("mutation", tuple(model.keys[idx] for idx in sorted(keys)),
                                            tuple(sorted(opened)))
        try:
            model.candidate_trace(candidate, state.domains)
        except ValueError:
            continue
        yield candidate


def _teacher_trace(model, domains, item, rng):
    candidate, metrics = item
    goals = metrics.get("pruned_outputs", candidate.open_colors)
    target = rng.choice(goals) if model.config.get("target_first") and goals else None
    return model.candidate_trace(candidate, domains, rng=rng, target_color=target)


def curriculum_prefix(model, domains, teachers, epoch, epochs, rng):
    """Learn outputs, then core completions, then complete unforced construction."""
    duration = max(1., .75 * epochs)
    if not teachers or epoch >= duration:
        return None
    trace = _teacher_trace(model, domains, rng.choice(teachers), rng)
    core_phase = trace.index(("finish", -1)) + 1
    return trace[:ceil(core_phase * (1 - epoch / duration))]


def teacher_loss(model, domains, teachers, rng, *, greedy=False):
    """Imitate measured constructions, optionally following the best supported path.

    The default credits total compatible probability along a random teacher
    trace. Greedy imitation concentrates on one executable continuation at each
    prefix; actions outside the finite teacher pool remain unverified.
    """
    trace = None if greedy else _teacher_trace(model, domains, rng.choice(teachers), rng)
    targets = [(set(model.key_idx[key] for key in item.core_keys),
                set(model.color_idx[color] for color in item.open_colors),
                set(model.color_idx[color] for color in metrics.get("pruned_outputs", item.open_colors)))
               for item, metrics in teachers]
    encoded = model.encode(domains)
    sizes = [len(domains[color]) for color in model.network["colors"]]
    keys, opened, output_phase, losses = [], [], False, []
    steps = len(trace) if trace is not None else model.config["max_cores"] + model.config["max_open_colors"] + 3
    for step in range(steps):
        actions, distribution = model.distribution(encoded, sizes, keys, opened, output_phase, 0.)
        valid = set()
        choosing_target = model.config.get("target_first") and not keys and not opened and not output_phase
        for cores, colors, goals in targets:
            if choosing_target:
                following = [("open", color) for color in goals]
            elif output_phase:
                if (cores != set(keys) or not set(opened) <= colors or
                        model.config.get("target_first") and opened[0] not in goals):
                    continue
                following = [("open", color) for color in colors - set(opened)]
            else:
                if not set(keys) <= cores or (model.config.get("target_first") and
                        (not set(opened) <= colors or opened[0] not in goals)):
                    continue
                following = [("core", key) for key in cores - set(keys)]
            valid.update(following or [("finish", -1)])
        indices = [idx for idx, item in enumerate(actions) if item in valid]
        if not indices:
            raise AssertionError("Measured teacher has no valid continuation")
        if greedy:
            choice = indices[int(distribution.logits[indices].detach().argmax())]
            action = actions[choice]
            losses.append(-distribution.logits[choice])
        else:
            action = trace[step]
            if action not in valid:
                raise AssertionError("Measured teacher has no valid continuation")
            losses.append(-distribution.logits[indices].logsumexp(0))
        kind, idx = action
        if kind == "core":
            keys.append(idx)
        elif kind == "open":
            opened.append(idx)
        elif not output_phase:
            output_phase = True
        elif greedy:
            break
    # The one starting decision must not vanish among many easier continuation
    # and output decisions as affordable clusters grow.
    return (losses[0] + torch.stack(losses[1:]).mean()) / 2


def completion_value_loss(model, domains, teachers):
    """Credit the best known terminal utility at the student's own prefixes.

    With exhaustive small-problem teachers these are exact completion labels;
    with sampled teachers they are lower bounds. A prefix that has left every
    measured completion gets no invented label. Its earlier mistaken choice
    was already supervised while useful completions remained reachable.
    """
    if not model.config.get("target_first"):
        raise ValueError("Completion-value imitation requires target-first construction")
    targets = [(set(model.key_idx[key] for key in candidate.core_keys),
                set(model.color_idx[color] for color in candidate.open_colors),
                set(model.color_idx[color] for color in metrics["pruned_outputs"]), metrics["reward"])
               for candidate, metrics in teachers]
    encoded = model.encode(domains)
    sizes = [len(domains[color]) for color in model.network["colors"]]
    keys, opened, phase, losses = [], [], False, []
    while True:
        actions, distribution = model.distribution(encoded, sizes, keys, opened, phase, 0.)
        values = {action: 0. for action in actions}
        for cores, outputs, goals, reward in targets:
            if not keys and not opened:
                following = [("open", color) for color in goals]
            elif not set(keys) <= cores or not set(opened) <= outputs or opened[0] not in goals:
                continue
            elif phase:
                if set(keys) != cores:
                    continue
                following = [("open", color) for color in outputs - set(opened)] or [("finish", -1)]
            else:
                following = [("core", key) for key in cores - set(keys)] or [("finish", -1)]
            for action in following:
                if action in values:
                    values[action] = max(values[action], reward)
        best = max(values.values())
        if best > 0:
            indices = [idx for idx, action in enumerate(actions)
                       if values[action] > 0 and values[action] >= best - .0001]
            losses.append(-distribution.logits[indices].logsumexp(0))
        kind, idx = actions[int(model._greedy_choice(actions, distribution))]
        if kind == "core":
            keys.append(idx)
        elif kind == "open":
            opened.append(idx)
        elif kind == "finish" and not phase:
            phase = True
        else:
            break
    if not losses:
        raise ValueError("Completion-value imitation requires a positive measured teacher")
    return (losses[0] + torch.stack(losses[1:]).mean()) / 2 if len(losses) > 1 else losses[0]


def on_policy_teacher_loss(model, state, teachers, *, work_budget=None):
    """Teach completion at student prefixes, retaining measured useful outputs.

    Adding cores to a useful teacher cannot create support on its retained
    outputs. This supplies affordable repairs without contracting each prefix.
    Student extensions are followed when a useful repair still fits; an
    irreversible mistake such as premature finish receives a correction.
    """
    if not model.config.get("target_first"):
        raise ValueError("On-policy imitation requires target-first construction")
    encoded = model.encode(state.domains)
    sizes = [len(state.domains[color]) for color in model.network["colors"]]
    targets = [(set(model.key_idx[key] for key in candidate.core_keys),
                set(model.color_idx[color] for color in candidate.open_colors),
                set(model.color_idx[color] for color in metrics.get("pruned_outputs", candidate.open_colors)))
               for candidate, metrics in teachers]
    checked = {}
    limits = {key: model.config[key] for key in ("max_work", "max_size")}
    if work_budget is not None:
        limits["max_work"] = min(limits["max_work"], work_budget)

    def completions(keys, opened, phase):
        result = []
        for cores, outputs, goals in targets:
            if not opened or opened[0] not in goals or phase and not cores <= set(keys):
                continue
            union, retained = set(keys) | cores, set(opened) | outputs
            signature = tuple(sorted(union)), tuple(sorted(retained))
            if signature not in checked:
                candidate = SummaryClusterCandidate("teacher_repair",
                    tuple(model.keys[idx] for idx in signature[0]),
                    tuple(model.network["colors"][idx] for idx in signature[1]))
                try:
                    model.candidate_trace(candidate, state.domains,
                                          target_color=model.network["colors"][opened[0]])
                    prepare_candidate(state, candidate,
                        compact_domains=model.config.get("compact_domains", False), **limits)
                except ValueError as error:
                    if str(error) not in (
                        "Candidate exceeds construction limits or is disconnected",
                        "Teacher outputs exceed the construction masks",
                        "Candidate cannot finish under the construction masks",
                        "Candidate exceeds the contraction work or tensor-size budget",
                        "NumPy contraction supports at most 52 distinct colors"):
                        raise
                    checked[signature] = False
                else:
                    checked[signature] = True
            if checked[signature]:
                result.append((union, retained))
        return result

    keys, opened, phase, losses = [], [], False, []
    while True:
        actions, distribution = model.distribution(encoded, sizes, keys, opened, phase, 0.)
        valid = set()
        if not keys and not opened:
            valid = {("open", goal) for _, _, goals in targets for goal in goals}
        else:
            for cores, outputs in completions(keys, opened, phase):
                remaining, kind = (outputs - set(opened), "open") if phase else (cores - set(keys), "core")
                valid.update([(kind, idx) for idx in remaining] or [("finish", -1)])
        indices = [idx for idx, action in enumerate(actions) if action in valid]
        if not indices:
            raise AssertionError("Student prefix has no affordable useful teacher completion")
        losses.append(-distribution.logits[indices].logsumexp(0))
        choice = int(model._greedy_choice(actions, distribution))
        action = actions[choice]
        if action not in valid:
            kind, idx = action
            repairable = (kind == "core" and completions([*keys, idx], opened, phase) or
                          kind == "open" and completions(keys, [*opened, idx], phase))
            if not repairable:
                choice = max(indices, key=lambda idx: float(distribution.logits[idx].detach()))
                action = actions[choice]
        kind, idx = action
        if kind == "core":
            keys.append(idx)
        elif kind == "open":
            opened.append(idx)
        elif not phase:
            phase = True
        else:
            break
    return (losses[0] + torch.stack(losses[1:]).mean()) / 2


def fit(data, *, output, epochs=12, width=32, rounds=2, lr=.0003, seed=0,
        puzzles_per_epoch=32, rollouts=3, steps=3, warmup=2, teacher_candidates=32,
        teacher_mutations=12, imitation_weight=1., entropy_weight=.001, cost_weight=.02,
        validation_seconds=5., max_total_work=5_000_000, train_indices=None,
        validation_indices=None, curriculum=True, initial_checkpoint=None, search_nodes=0,
        search_seconds=10., on_policy_imitation=False, transfer_weights=False, teacher_probes=0,
        completion_value_imitation=False, greedy_imitation=False, objective="mixed", cluster_attempts=32,
        device="cpu", validation_interval=1, checkpoint_every=0, progress_every=100,
        teacher_cache_size=256, cycle_puzzles=False, teacher_max_steps=None, nested_teacher=False, **limits):
    """Positive measured-trace warm-up, then immediate REINFORCE on fresh samples."""
    output = Path(output)
    paths = [output, output.with_stem(output.stem + ".best"), output.with_stem(output.stem + ".initial")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Output already exists; choose a fresh experiment path")
    teacher_max_steps = _teacher_search_steps(teacher_candidates, teacher_max_steps, nested_teacher)
    if (min(epochs, width, rounds, lr, steps, teacher_candidates) <= 0 or rollouts < 2
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                   for value in (puzzles_per_epoch, teacher_cache_size))
            or not isinstance(progress_every, int) or isinstance(progress_every, bool) or progress_every < 1
            or not isinstance(cycle_puzzles, bool) or not isinstance(greedy_imitation, bool)
            or max_total_work is not None and (not isinstance(max_total_work, int)
                or isinstance(max_total_work, bool) or max_total_work < 1)
            or not isinstance(cluster_attempts, int) or isinstance(cluster_attempts, bool) or cluster_attempts < 1
            or not isinstance(search_nodes, int) or isinstance(search_nodes, bool) or search_nodes < 0
            or not isinstance(teacher_probes, int) or isinstance(teacher_probes, bool) or teacher_probes < 0
            or not isinstance(validation_interval, int) or isinstance(validation_interval, bool) or validation_interval < 1
            or not isinstance(checkpoint_every, int) or isinstance(checkpoint_every, bool) or checkpoint_every < 0
            or warmup < 0 or teacher_mutations < 0 or validation_seconds <= 0
            or search_seconds <= 0
            or any(not isfinite(value) or value < 0 for value in
                   (lr, imitation_weight, entropy_weight, cost_weight, validation_seconds, search_seconds))):
        raise ValueError("Invalid learning parameters")
    if objective not in {"mixed", "dfs_nodes", "pruning"} or objective == "dfs_nodes" and not search_nodes:
        raise ValueError("The dfs_nodes objective requires positive --search-nodes")
    training_device = torch.device(device)
    if training_device.type not in {"cpu", "cuda"}:
        raise ValueError("Construction learning supports CPU or CUDA training")
    if training_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable: use CUDA-enabled PyTorch on an allocated GPU")
    snapshots = {epoch: output.with_stem(output.stem + f".epoch{epoch:04d}")
                 for epoch in range(checkpoint_every, epochs + 1, checkpoint_every)} if checkpoint_every else {}
    if any(path.exists() for path in snapshots.values()):
        raise FileExistsError("Epoch snapshot already exists; choose a fresh experiment path")
    if objective == "dfs_nodes" and on_policy_imitation:
        raise ValueError("DFS-node teachers require measured completions; repair imitation is ungraded")
    if on_policy_imitation and not limits.get("target_first"):
        raise ValueError("On-policy imitation requires target-first construction")
    if completion_value_imitation and (on_policy_imitation or not limits.get("target_first")):
        raise ValueError("Completion-value imitation requires target-first construction and excludes repair imitation")
    if greedy_imitation and (on_policy_imitation or completion_value_imitation):
        raise ValueError("Greedy imitation excludes repair and completion-value imitation")
    if transfer_weights and (initial_checkpoint is None or not limits.get("value_encoding")):
        raise ValueError("Cross-network transfer requires value encoding and an initialization checkpoint")
    base, state, pools, validation = training_states(data, train_indices, validation_indices, progress_every)
    if objective == "dfs_nodes" and not validation:
        raise ValueError("The dfs_nodes objective requires validation instances for checkpoint selection")
    torch.manual_seed(seed)
    rng = Random(seed)
    config = dict(max_cores=32, max_open_colors=6, compact_domains=True, contextual=True,
                  heuristic_candidates=32, max_total_work=max_total_work, min_active=2, cyclic_only=True,
                  cluster_attempts=cluster_attempts)
    config.update(limits)
    model = Policy(base["network"], width, rounds, **config)
    model.reward_colors, model.beta = base["reward_colors"], 0.
    parent, inherited = None, set()
    initial_domains, _ = instances(data)
    exposure = _instance_hashes(data, [initial_domains[idx] for idx in pools])
    if initial_checkpoint is not None:
        initial_checkpoint = Path(initial_checkpoint)
        saved = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
        if (not saved.get("config", {}).get("contextual") or
                saved.get("network") != model.network and not transfer_weights):
            raise ValueError("Initialization requires a contextual policy for the same tensor network")
        if transfer_weights and not saved.get("config", {}).get("value_encoding"):
            raise ValueError("Cross-network transfer requires a value-encoded initialization checkpoint")
        inherited = set(saved.get("training", {}).get("train_puzzles", ()))
        inherited_hashes = saved.get("training", {}).get("train_instance_hashes")
        if inherited_hashes is None:
            raise ValueError("Initialization checkpoint lacks training instance fingerprints")
        if set(inherited_hashes) & _instance_hashes(data, validation):
            raise ValueError("Initialization checkpoint has trained on validation puzzles")
        exposure.update(inherited_hashes)
        model.load_state_dict(saved["weights"])
        parent = dict(path=str(initial_checkpoint), sha256=sha256(initial_checkpoint.read_bytes()).hexdigest())
        if saved.get("network") != model.network:
            parent["cross_network"] = True
    model.to(training_device)
    output.parent.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = ClusterCandidateGenerator(state.constraint_network, max_candidates=teacher_candidates,
        max_steps=teacher_max_steps,
        **{key: model.config[key] for key in ("max_cores", "max_open_colors", "max_work", "max_size")})
    generate = generator.generate_nested if nested_teacher else generator.generate
    cache, history, second_moment = {}, [], .01
    search_statistics = dict(calls=0, seconds=0., limited=0)

    @lru_cache(maxsize=1024)
    def search_snapshot(snapshot):
        row = solve(state.constraint_network, dict(zip(model.network["colors"], snapshot)),
                    "plain", model, nodes=search_nodes, seconds=search_seconds)
        search_statistics["calls"] += 1
        search_statistics["seconds"] += row["setup_s"] + row["solve_s"]
        search_statistics["limited"] += int(row["status"] not in {"solved", "unsat"})
        return {key: row[key] for key in ("status", "nodes")}

    def search_domains(domains):
        return search_snapshot(tuple(tuple(domains[color]) for color in model.network["colors"]))

    # Node budgets still enable full solver validation, but pruning training
    # never grades every sampled construction by counterfactual DFS searches.
    search = search_domains if search_nodes and objective != "pruning" else None

    def oracle():
        snapshot = tuple(state.domains.values()), remaining_work
        if snapshot in cache:
            positives, structural, measured = cache[snapshot]
            return positives, 0, structural, measured
        proposals = list(generate(state.domains, cyclic_only=True,
                                  compact_domains=model.config["compact_domains"]))
        totals["teacher_structural_calls"] += 1
        totals["teacher_structural_candidates"] += len(proposals)
        structural_proposals = {_signature(candidate) for candidate in proposals}
        if teacher_probes:
            began = perf_counter()
            proposals.extend(causal_candidates(state, model.reward_colors, rng, teacher_probes,
                                               max_cores=model.config["max_cores"]))
            totals["teacher_probe_seconds"] += perf_counter() - began
            totals["teacher_probe_candidates"] += len(proposals) - len(structural_proposals)
            totals["teacher_probe_assumptions"] += min(teacher_probes, sum(len(state.domains[color])
                for color in model.reward_colors if len(state.domains[color]) > 1))
        scored, signatures = [], set()
        with torch.no_grad():
            for candidate in proposals:
                if _signature(candidate) in signatures:
                    continue
                try:
                    model.candidate_trace(candidate, state.domains)
                except ValueError:
                    continue
                metrics = measure(model, state, candidate, cost_weight=cost_weight,
                                  known_satisfiable=sample["known_satisfiable"], work_budget=remaining_work,
                                  search=search, objective=objective)
                signatures.add(_signature(candidate))
                scored.append((candidate, metrics))
            structural = signatures & structural_proposals
            totals["teacher_structural_useful"] += sum(metrics["useful"] for candidate, metrics in scored
                if _signature(candidate) in structural)
            seeds = [candidate for candidate, metrics in scored if metrics["useful"]] or proposals
            if seeds:
                for candidate in mutations(model, state, seeds, rng, teacher_mutations):
                    signature = _signature(candidate)
                    if signature not in signatures:
                        metrics = measure(model, state, candidate, cost_weight=cost_weight,
                                          known_satisfiable=sample["known_satisfiable"], work_budget=remaining_work,
                                          search=search, objective=objective)
                        scored.append((candidate, {**metrics, "novel": 1}))
                        signatures.add(signature)
        positives = _positive_teachers(scored, search is not None, objective=objective)
        if teacher_cache_size and len(cache) >= teacher_cache_size:
            cache.pop(next(iter(cache)))
        measured = set(signatures)
        cache[snapshot] = positives, structural, measured
        return positives, len(scored), structural, measured

    def selection():
        model.eval()
        # Runtime loads policies on CPU; keep validation decisions consistent
        # with deployment while the learner and its optimizer remain on GPU.
        validator = model if training_device.type == "cpu" else deepcopy(model).cpu().eval()
        if validation:
            print(f"validation root propagation: {len(validation)} puzzles", flush=True)
            result = evaluate_roots(state.constraint_network, validation, validator,
                                    seconds=validation_seconds, total_work=max_total_work,
                                    cluster_attempts=cluster_attempts)
            compact = {key: value for key, value in result.items() if key != "rows"}
            if compact["contradictions"] and "puzzles" in data:
                raise AssertionError("Validation contraction contradicted a known satisfiable puzzle")
            print(f"validation greedy constructions: {len(validation)} puzzles", flush=True)
            greedy = []
            for domains in validation:
                probe = state.constraint_network.create_mac_state(domains)
                if probe.propagate() and not probe.is_solved():
                    greedy.append(measure(validator, probe, validator.propose(probe.domains),
                                          objective="pruning"))
            compact.update(greedy_trials=len(greedy), greedy_useful=sum(row["useful"] for row in greedy),
                           greedy_reward=sum(row["reward"] for row in greedy) / max(1, len(greedy)))
            rank = (compact["additional_resolved_fraction"], compact["mean_progress"], -compact["mean_seconds"])
            if search_nodes:
                print(f"validation DFS: {len(validation)} puzzles", flush=True)
                rows = []
                for index, domains in enumerate(validation, 1):
                    row = solve(state.constraint_network, domains, "trained", validator,
                                nodes=search_nodes, seconds=validation_seconds, total_work=max_total_work,
                                cluster_attempts=cluster_attempts)
                    rows.append(row)
                    print(f"validation DFS {index}/{len(validation)}: {row['status']}, nodes {row['nodes']}",
                          flush=True)
                if any(row["status"] == "invalid" or "puzzles" in data and row["status"] == "unsat"
                       for row in rows):
                    raise AssertionError("Invalid validation search outcome")
                complete = sum(row["status"] in {"solved", "unsat"} for row in rows)
                nodes = sum(row["nodes"] or 0 for row in rows)
                compact.update(search_complete=complete,
                               search_nodes=nodes if complete == len(rows) else None,
                               search_statuses=[row["status"] for row in rows],
                               search_nodes_by_instance=[row["nodes"] for row in rows],
                               search_seconds=sum(row["solve_s"] for row in rows))
                # Unknown node counts must never make a timeout look cheaper.
                rank = (complete, -nodes if complete == len(rows) else -float("inf"),
                        *(rank if objective == "mixed" else ()))
            return dict(source="validation_roots", **compact), rank
        return dict(source="unvalidated", states=0), (0., 0., 0.)

    epoch_puzzles = min(puzzles_per_epoch or len(pools), len(pools))
    chunks = ceil(len(pools) / epoch_puzzles)
    epoch_unit = "chunk" if cycle_puzzles and chunks > 1 else "epoch"
    schedule = ("full pass" if chunks == 1 else f"{chunks} chunks per full pass" if cycle_puzzles else "sampled pass")
    planned_visits = ((epochs // chunks) * len(pools) + min(epochs % chunks * epoch_puzzles, len(pools))
                      if cycle_puzzles else epochs * epoch_puzzles)
    progress = dict(puzzle_visits=0, state_exposures=0, updates=0, puzzle_passes=0., unique_puzzles_visited=0)
    visited_puzzles = set()
    print(f"Construction learning on {training_device}: {objective}; {warmup} teacher-only {epoch_unit}s, "
          f"then REINFORCE plus imitation weight {imitation_weight:g}", flush=True)
    print(f"training data: {len(initial_domains)} dataset instances including {len(validation)} validation; "
          f"{len(pools)} unresolved training puzzles, {sum(map(len, pools.values()))} contexts; "
          f"up to {epoch_puzzles} puzzles per {epoch_unit} ({schedule})", flush=True)
    print(f"structural teacher: {'nested' if nested_teacher else 'direct'}, up to {teacher_candidates} proposals; "
          f"largest-stage search budget {teacher_max_steps} steps", flush=True)
    if training_device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(training_device)}; PyTorch {torch.__version__}; "
              f"CUDA {torch.version.cuda}; validation on CPU", flush=True)
    print("validating initialization", flush=True)
    initial, best = selection()
    last_validation_epoch = 0
    saved_cost = cost_weight if objective == "mixed" else 0.
    saved_contradiction_reward = 0. if objective == "dfs_nodes" else 1.
    cost_metric = {"mixed": "conditioned_contraction_work", "dfs_nodes": "dfs_nodes_saved", "pruning": "none"}[objective]
    training_metadata = dict(method="immediate_construction", seed=seed,
        train_puzzles=sorted(inherited | set(pools)), train_instance_hashes=sorted(exposure), parent=parent,
        objective=objective, reward_units={"dfs_nodes": "nodes_saved", "pruning": "entropy_fraction", "mixed": "mixed"}[objective],
        cluster_attempts=cluster_attempts, max_total_work=max_total_work,
        search_nodes=search_nodes, search_seconds=search_seconds,
        device=str(training_device), validation_device="cpu", lr=lr, epochs=epochs,
        planned_epochs=epochs, warmup=warmup, greedy_imitation=greedy_imitation,
        puzzles_per_epoch=puzzles_per_epoch, puzzles_per_epoch_effective=epoch_puzzles,
        dataset_instances=len(initial_domains), validation_instances=len(validation),
        train_puzzle_count=len(pools), train_context_count_initial=sum(map(len, pools.values())),
        train_context_count=sum(map(len, pools.values())),
        planned_puzzle_visits=planned_visits, planned_puzzle_passes=planned_visits / len(pools),
        cycle_puzzles=cycle_puzzles, epoch_unit=epoch_unit, chunks_per_pass=chunks if cycle_puzzles else None,
        progress_every=progress_every,
        teacher_candidates=teacher_candidates, teacher_max_steps=teacher_max_steps, nested_teacher=nested_teacher,
        teacher_cache_size=teacher_cache_size, **progress,
        data_sha256=sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        validation_seconds=validation_seconds, validation_interval=validation_interval,
        checkpoint_every=checkpoint_every, epoch=0, last_validation_epoch=0)
    for path in paths[1:]:
        _save(path, model, [], saved_cost, saved_contradiction_reward, initial,
              cost_metric=cost_metric, training=training_metadata)
    pending_puzzles = []
    for epoch in range(epochs):
        model.train()
        if not cycle_puzzles or not pending_puzzles:
            pending_puzzles = list(pools)
            rng.shuffle(pending_puzzles)
        puzzle_order = pending_puzzles[:epoch_puzzles]
        pending_puzzles = pending_puzzles[epoch_puzzles:] if cycle_puzzles else []
        totals = defaultdict(float, updates=0., state_exposures=0., puzzle_visits=0.,
                             teacher_structural_calls=0., teacher_structural_candidates=0., teacher_structural_useful=0.)
        started = last_progress = perf_counter()
        phase = "teacher_only" if epoch < warmup else "reinforce_and_imitation"
        print(f"{epoch_unit} {epoch + 1}/{epochs} ({phase}): starting {len(puzzle_order)} puzzles", flush=True)
        for puzzle_number, puzzle in enumerate(puzzle_order, 1):
            totals["puzzle_visits"] += 1
            progress["puzzle_visits"] += 1
            visited_puzzles.add(puzzle)
            roots = [item for item in pools[puzzle] if item["root"]]
            sample = roots[0] if roots and rng.random() < .5 else rng.choice(pools[puzzle])
            with _closed(state, sample, model.network["colors"]):
                remaining_work = max_total_work
                for _ in range(steps):
                    if remaining_work is not None and remaining_work < 1:
                        break
                    totals["state_exposures"] += 1
                    progress["state_exposures"] += 1
                    teachers, trials, structural, measured_candidates = oracle()
                    totals["teacher_trials"] += trials
                    totals["positive_teachers"] += len(teachers)
                    optimizer.zero_grad()
                    sampled = []
                    prefix = curriculum_prefix(model, state.domains, teachers, epoch, epochs, rng) if curriculum else None
                    for _ in range(rollouts):
                        began = perf_counter()
                        candidate, log_prob, entropy, trace = model.rollout(
                            state.domains, beta=1. if epoch < warmup else 0., prefix=prefix)
                        metrics = measure(model, state, candidate, cost_weight=cost_weight,
                            proposal_s=perf_counter() - began, known_satisfiable=sample["known_satisfiable"],
                            work_budget=remaining_work, search=search, objective=objective)
                        signature = None if candidate is None else _signature(candidate)
                        metrics.update(prefixed=int(bool(prefix)),
                            novel=int(candidate is not None and signature not in structural),
                            useful_novel=int(metrics["useful"] and signature not in structural),
                            useful_unseen=int(metrics["useful"] and signature not in measured_candidates),
                            useful_unforced=int(metrics["useful"] and not prefix))
                        sampled.append((candidate, log_prob, entropy, metrics))
                    labeled = [item for item in sampled if item[3]["reward_valid"]]
                    rewards = [item[3]["reward"] for item in labeled]
                    scale = max(.05, sqrt(second_moment))
                    losses = [-((rewards[idx] - ((sum(rewards) - rewards[idx]) / (len(rewards) - 1)
                               if len(rewards) > 1 else 0.)) / scale) * log_prob - entropy_weight * entropy
                              for idx, (_, log_prob, entropy, _) in enumerate(labeled)]
                    policy_loss = torch.stack(losses).mean() if losses else model.head[-1].weight.sum() * 0
                    imitation = policy_loss * 0
                    if teachers:
                        imitation = (on_policy_teacher_loss(model, state, teachers, work_budget=remaining_work)
                                     if on_policy_imitation else teacher_loss(model, state.domains, teachers, rng,
                                                                            greedy=greedy_imitation))
                        if completion_value_imitation:
                            imitation = (imitation + completion_value_loss(model, state.domains, teachers)) / 2
                    loss = (0. if epoch < warmup else 1.) * policy_loss + imitation_weight * imitation
                    if (objective == "mixed" or epoch >= warmup and labeled
                            or teachers and imitation_weight > 0):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                        totals["updates"] += 1
                        progress["updates"] += 1
                    if rewards:
                        second_moment = .95 * second_moment + .05 * sum(value ** 2 for value in rewards) / len(rewards)
                    totals["imitation_loss"] += float(imitation.detach())
                    totals["policy_loss"] += float(policy_loss.detach())
                    totals["reward_scale"] += scale
                    for _, _, _, metrics in sampled:
                        totals["trials"] += 1
                        for key, value in metrics.items():
                            if key != "pruned_outputs":
                                totals[key] += value
                    useful = [(candidate, metrics) for candidate, _, _, metrics in sampled if metrics["useful"]]
                    # Reuse successful generated constructions as measured teachers
                    # when this exact state is revisited, without fixed proposals at inference.
                    replay = {_signature(candidate): (candidate, metrics) for candidate, metrics in teachers}
                    replay.update({_signature(candidate): (candidate, metrics) for candidate, metrics in useful
                                   if metrics["reward"] > 0})
                    cache[tuple(state.domains.values()), remaining_work] = (
                        _positive_teachers(replay.values(), search is not None, objective=objective)[:64],
                        structural, measured_candidates | set(replay))
                    positive_sampled = _positive_teachers(useful, search is not None, objective=objective)
                    next_choices = positive_sampled or teachers[:1]
                    if not next_choices:
                        break
                    chosen, measured = max(next_choices, key=lambda item: item[1]["reward"])
                    result = apply_candidate(state, chosen,
                        compact_domains=model.config["compact_domains"],
                        max_work=(model.config["max_work"] if remaining_work is None
                                  else min(model.config["max_work"], remaining_work)), max_size=model.config["max_size"])
                    if remaining_work is not None:
                        remaining_work -= measured["work"]
                    totals["teacher_advances"] += int(not positive_sampled)
                    if result.contradiction or state.is_solved() or not result.removed_values:
                        break
                    snapshot = tuple(state.domains.values())
                    if len(pools[puzzle]) < 12 and all(item["domains"] != snapshot for item in pools[puzzle]):
                        pools[puzzle].append(dict(domains=snapshot, root=False,
                                                 known_satisfiable=sample["known_satisfiable"]))
            if puzzle_number % progress_every == 0 or perf_counter() - last_progress >= 60:
                print(f"{epoch_unit} {epoch + 1}/{epochs} ({phase}): puzzles {puzzle_number}/{len(puzzle_order)}, "
                      f"states {int(totals['state_exposures'])}, updates {int(totals['updates'])} "
                      f"({progress['updates']} total), sampled useful {int(totals['useful'])}/{int(totals['trials'])}, "
                      f"imitation {totals['imitation_loss'] / max(1, totals['state_exposures']):.3f}, "
                      f"passes {progress['puzzle_visits'] / len(pools):.2f}", flush=True)
                last_progress = perf_counter()
        progress.update(puzzle_passes=progress["puzzle_visits"] / len(pools),
                        unique_puzzles_visited=len(visited_puzzles))
        validated = (epoch + 1) % validation_interval == 0 or epoch + 1 == epochs
        validation_result, rank = selection() if validated else (None, None)
        if validated:
            last_validation_epoch = epoch + 1
        row = dict(epoch=epoch + 1, phase=phase, seconds=perf_counter() - started, **totals, validation=validation_result,
                   total_updates=progress["updates"], total_puzzle_visits=progress["puzzle_visits"],
                   total_state_exposures=progress["state_exposures"], puzzle_passes=progress["puzzle_passes"],
                   train_context_count=sum(map(len, pools.values())),
                   search_grading=dict(search_statistics))
        history.append(row)
        metadata = dict(training_metadata, **progress,
                        train_context_count=sum(map(len, pools.values())),
                        initial_validation=initial, rollouts=rollouts, warmup=warmup,
                        imitation_weight=imitation_weight, teacher_candidates=teacher_candidates,
                        teacher_probes=teacher_probes,
                        teacher_mutations=teacher_mutations, curriculum=curriculum, parent=parent,
                        teacher_objective="greedy_supported_action" if greedy_imitation else "legal_continuation_mass",
                        target_labels="pruned_outputs",
                        on_policy_imitation=on_policy_imitation,
                        completion_value_imitation=completion_value_imitation,
                        greedy_imitation=greedy_imitation,
                        transfer_weights=transfer_weights,
                        epoch=epoch + 1, phase=phase, last_validation_epoch=last_validation_epoch,
                        search_grading=dict(search_statistics))
        if validated and (best is None or rank > best):
            _save(paths[1], model, history, saved_cost, saved_contradiction_reward, validation_result,
                  cost_metric=cost_metric, training=metadata)
            best = rank
        selection_record = validation_result if validated else dict(source="skipped", epoch=epoch + 1,
                                                                   last_validation_epoch=last_validation_epoch)
        _save(output, model, history, saved_cost, saved_contradiction_reward, selection_record,
              cost_metric=cost_metric, training=metadata)
        if epoch + 1 in snapshots:
            path = snapshots[epoch + 1]
            if path.exists():
                raise FileExistsError(f"Epoch snapshot already exists: {path}")
            _save(path, model, history, saved_cost, saved_contradiction_reward, selection_record,
                  cost_metric=cost_metric, training=metadata)
        validation_text = (f"greedy useful {validation_result.get('greedy_useful', 0)}/"
                           f"{validation_result.get('greedy_trials', 0)}, "
                           f"validation {validation_result.get('additional_solved', 0)} additional solved, "
                           f"progress {validation_result.get('mean_progress', 0):.4f}, "
                           f"DFS nodes {validation_result.get('search_nodes', 'ungraded')}"
                           if validated else "validation skipped")
        search_text = str(int(totals["search_improved"])) if search is not None else "ungraded"
        print(f"{epoch_unit} {epoch + 1}/{epochs} ({phase}): sampled useful {int(totals['useful'])}/{int(totals['trials'])}, "
              f"imitation {totals['imitation_loss'] / max(1, totals['updates']):.3f}, "
              f"policy loss {totals['policy_loss'] / max(1, totals['updates']):.3f}, {validation_text}, "
              f"valid labels {int(totals['reward_valid'])}/{int(totals['trials'])}, "
              f"node-saving trials {search_text}, puzzles {len(puzzle_order)}/{len(pools)}, "
              f"states {int(totals['state_exposures'])}, updates {int(totals['updates'])} "
              f"({progress['updates']} total), passes {progress['puzzle_passes']:.2f}, "
              f"structural proposals {int(totals['teacher_structural_candidates'])} on "
              f"{int(totals['teacher_structural_calls'])} fresh states "
              f"({int(totals['teacher_structural_useful'])} useful)", flush=True)
    return model.eval(), history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    for key, default in dict(epochs=12, width=32, rounds=2, lr=.0003, seed=0, puzzles_per_epoch=32,
                            rollouts=3, steps=3, warmup=2, teacher_candidates=32, teacher_mutations=12, teacher_probes=0,
                            imitation_weight=1., validation_seconds=5., max_total_work=5_000_000,
                            max_cores=32, max_open_colors=6, max_work=1_000_000, max_size=100_000,
                            min_active=2, search_nodes=0, search_seconds=10., cluster_attempts=32,
                            validation_interval=1, checkpoint_every=0, progress_every=100,
                            teacher_cache_size=256).items():
        help_text = {"puzzles_per_epoch": "Puzzles per epoch; 0 visits every unresolved training puzzle once",
                     "teacher_cache_size": "Cached measured teacher states; 0 retains all encountered states",
                     "progress_every": "Print training progress after this many puzzles (also every 60 seconds)"}
        parser.add_argument("--" + key.replace("_", "-"), type=type(default), default=default,
                            help=help_text.get(key))
    parser.add_argument("--objective", choices=("mixed", "dfs_nodes", "pruning"), default="mixed")
    parser.add_argument("--decoder", choices=("depth_first", "discrepancy", "sampling", "sampling_grow"), default="depth_first",
                        help="Actor decoding; sampling_grow extends while legal additions remain")
    parser.add_argument("--nested-teacher", action="store_true",
                        help="Keep structural proposal prefixes while expanding doubling stages from 32 candidates")
    parser.add_argument("--teacher-max-steps", type=int,
                        help="Largest structural search-stage budget; default 8192, or 256 per rounded-up candidate in nested mode")
    parser.add_argument("--device", default="cpu", help="cpu or cuda; CUDA uses HTCondor's visible GPU")
    parser.add_argument("--no-total-work-limit", action="store_const", const=None, dest="max_total_work",
                        help="Keep per-cluster affordability caps but remove the cumulative contraction-work cap")
    parser.add_argument("--train-indices", type=int, nargs="+")
    parser.add_argument("--validation-indices", type=int, nargs="+")
    parser.add_argument("--no-curriculum", action="store_false", dest="curriculum")
    parser.add_argument("--cycle-puzzles", action="store_true",
                        help="Consume a shuffled dataset in epoch-sized chunks before repeating any puzzle")
    parser.add_argument("--target-first", action="store_true", help="Choose an open variable before constructing its cluster")
    parser.add_argument("--on-policy-imitation", action="store_true", help="Teach affordable useful completions at student prefixes")
    parser.add_argument("--greedy-imitation", action="store_true",
                        help="Train the highest-scoring teacher-compatible continuation at corrected greedy prefixes")
    parser.add_argument("--completion-value-imitation", action="store_true", help="Also teach best-known terminal utility at student prefixes")
    parser.add_argument("--value-encoding", action="store_true", help="Encode allowed tensor assignments without absolute value labels")
    parser.add_argument("--relational-context", action="store_true", help="Retain selected tensor-value correspondences using target-value queries")
    parser.add_argument("--relational-rounds", type=int, default=6, help="Selected value/tuple message rounds for relational context")
    parser.add_argument("--initial-checkpoint", type=Path, help="Initialize matching contextual weights; use a new output")
    parser.add_argument("--transfer-weights", action="store_true", help="Explicitly allow value-encoded weights from another network")
    args = vars(parser.parse_args())
    path = args.pop("data")
    torch.set_num_threads(1)
    fit(json.loads(path.read_text()), **args)


if __name__ == "__main__":
    main()
