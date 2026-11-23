"""Fixed-teacher overfit and repair-imitation diagnostic on development data."""
import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from time import perf_counter

import torch

from ..candidate_evaluation_tnreason import SummaryClusterCandidate
from ..candidate_generation import ClusterCandidateGenerator
from . import learn
from .experiment import fingerprints, instances, propagate, solve
from .model import load
from .train import _closed, _save


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, separators=(",", ":"))


def freeze(data, model, *, count, seed):
    """Select by puzzle before measuring; validation never enters training."""
    _, state, pools, validation = learn.training_states(data)
    initial, _ = instances(data)
    if fingerprints(data, [initial[i] for i in data["train_indices"]]) & fingerprints(data, validation):
        raise ValueError("Training and validation puzzles overlap")
    rng = Random(seed)
    order = sorted(pools)
    rng.shuffle(order)
    chosen = []
    for n, puzzle in enumerate(order[:count]):
        branches = [s for s in pools[puzzle] if not s["root"]]
        roots = [s for s in pools[puzzle] if s["root"]]
        sample = rng.choice((branches if n % 2 else roots) or pools[puzzle])
        chosen.append(dict(sample, puzzle=puzzle, split="train"))
    for puzzle, domains in zip(data["validation_indices"], validation):
        state.push_level()
        try:
            for color, values in domains.items():
                state.restrict_domain(color, values)
            if state.propagate() and not state.is_solved():
                chosen.append(dict(domains=list(state.domains.values()), puzzle=puzzle,
                                   root=True, known_satisfiable=True, split="validation"))
        finally:
            state.pop_level()
    generator = ClusterCandidateGenerator(state.constraint_network, max_candidates=64,
        **{k: model.config[k] for k in ("max_cores", "max_open_colors", "max_work", "max_size")})
    records = []
    for idx, sample in enumerate(chosen):
        with _closed(state, sample, model.network["colors"]):
            before = state.snapshot_domains()
            proposals = list(generator.generate(state.domains, cyclic_only=True, compact_domains=True))
            proposals += learn.causal_candidates(state, model.reward_colors, Random(seed + idx), 80,
                                                max_cores=model.config["max_cores"])
            scored, seen = [], set()
            for candidate in proposals:
                signature = learn._signature(candidate)
                if signature in seen:
                    continue
                seen.add(signature)
                try:
                    model.candidate_trace(candidate, state.domains)
                except ValueError:
                    continue
                metrics = learn.measure(model, state, candidate, objective="pruning",
                                        known_satisfiable=sample["known_satisfiable"])
                if metrics["useful"] and metrics["reward"] > 0:
                    scored.append((candidate, metrics))
            teachers = learn._positive_teachers(scored, objective="pruning")[:8]
            if before != state.snapshot_domains():
                raise AssertionError("Teacher measurement changed the source state")
        records.append({**sample, "id": idx, "tested": len(seen), "teachers": [
            dict(cores=c.core_keys, outputs=c.open_colors, metrics=m) for c, m in teachers]})
        print("freeze", sample["split"], sample["puzzle"], "teachers", len(teachers), flush=True)
    return state, records, validation


def teachers_for(record):
    return [(SummaryClusterCandidate("fixed", tuple(t["cores"]), tuple(t["outputs"])), t["metrics"])
            for t in record["teachers"]]


def departure(model, domains, teachers):
    """First greedy departure from finite teacher support, not a uselessness label."""
    targets = [(set(model.key_idx[k] for k in c.core_keys),
                set(model.color_idx[c] for c in c.open_colors),
                set(model.color_idx[c] for c in m["pruned_outputs"])) for c, m in teachers]
    original = model.distribution
    first = None

    def traced(encoded, sizes, keys, opened, phase, *args):
        nonlocal first
        actions, distribution = original(encoded, sizes, keys, opened, phase, *args)
        valid = set()
        for cores, outputs, goals in targets:
            if not keys and not opened and not phase:
                following = [("open", c) for c in goals]
            elif not set(keys) <= cores or not set(opened) <= outputs or opened[0] not in goals:
                continue
            elif phase:
                if cores != set(keys):
                    continue
                following = [("open", c) for c in outputs - set(opened)]
            else:
                following = [("core", k) for k in cores - set(keys)]
            valid.update(following or [("finish", -1)])
        selected = int(model._greedy_choice(actions, distribution))
        if first is None and actions[selected] not in valid:
            indices = [i for i, a in enumerate(actions) if a in valid]
            first = dict(stage="target" if not keys and not opened else "outputs" if phase else "cores",
                         action=actions[selected], cores=len(keys), outputs=len(opened),
                         compatible_mass=float(distribution.probs[indices].sum()),
                         compatible_best=float(distribution.probs[indices].max()) if indices else 0.,
                         chosen_probability=float(distribution.probs[selected]))
        return actions, distribution

    model.distribution = traced
    try:
        candidate, _, _, trace = model.rollout(domains, beta=0., greedy=True)
    finally:
        del model.__dict__["distribution"]
    return candidate, first, trace


def probe(model, state, records, seed):
    model.eval()
    rows = []
    with torch.no_grad():
        for record in records:
            with _closed(state, record, model.network["colors"]):
                teachers = teachers_for(record)
                loss = float(learn.teacher_loss(model, state.domains, teachers, Random(seed + record["id"]))) if teachers else None
                candidate, first, trace = departure(model, state.domains, teachers)
                metrics = learn.measure(model, state, candidate, objective="pruning",
                                        known_satisfiable=record["known_satisfiable"])
                rows.append(dict(id=record["id"], puzzle=record["puzzle"], split=record["split"],
                    has_teacher=bool(teachers), loss=loss, departure=first, steps=len(trace),
                    teacher_match=candidate is not None and learn._signature(candidate) in {
                        learn._signature(c) for c, _ in teachers}, metrics=metrics))
    groups = {}
    for split in ("train", "validation"):
        selected = [r for r in rows if r["split"] == split]
        labeled = [r for r in selected if r["has_teacher"]]
        groups[split] = dict(states=len(selected), teacher_states=len(labeled),
            loss=sum(r["loss"] for r in labeled) / len(labeled) if labeled else None,
            useful=sum(r["metrics"]["useful"] for r in selected),
            useful_with_teacher=sum(r["metrics"]["useful"] for r in labeled),
            reward=sum(r["metrics"]["reward"] for r in selected) / max(1, len(selected)),
            teacher_matches=sum(r["teacher_match"] for r in selected),
            departures=dict(Counter(r["departure"]["stage"] + ":" + r["departure"]["action"][0]
                                    for r in labeled if r["departure"])))
    return dict(groups=groups, rows=rows)


def solver_probe(model, state, validation, *, mode):
    rows = []
    for domains in validation:
        root = propagate(state.constraint_network, domains, mode, model, seconds=120.,
                         cluster_attempts=32, baseline_candidates=32)
        row = solve(state.constraint_network, domains, mode, model, nodes=10000, seconds=120.,
                    cluster_attempts=32, baseline_candidates=32)
        if row["status"] in {"invalid", "unsat"}:
            raise AssertionError("Invalid solution on a satisfiable validation puzzle")
        row.pop("assignment", None)
        row["root"] = {k: root[k] for k in ("solved", "baseline_solved", "progress", "work")}
        rows.append(row)
    complete = all(r["status"] == "solved" for r in rows)
    return dict(complete=complete, nodes=sum(r["nodes"] for r in rows) if complete else None,
                additional_roots=sum(r["root"]["solved"] and not r["root"]["baseline_solved"] for r in rows), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--states", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    if min(args.states, args.epochs) < 1 or args.output.exists():
        parser.error("Use positive counts and a new output directory")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    data = json.loads(args.data.read_text())
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    inherited = saved.get("training", {}).get("train_instance_hashes")
    initial, _ = instances(data)
    if inherited is None or set(inherited) & learn._instance_hashes(
            data, [initial[i] for i in data["validation_indices"]]):
        raise ValueError("Parent lacks exposure metadata or has trained on validation puzzles")
    parent = load(args.checkpoint)
    if parent.network != data["network"] or not parent.config.get("target_first"):
        raise ValueError("Expected a matching target-first construction policy")
    parent.beta = 0.
    parent.config.pop("max_total_work", None)
    args.output.mkdir(parents=True)
    state, records, validation = freeze(data, parent, count=args.states, seed=args.seed)
    training = [r for r in records if r["split"] == "train" and r["teachers"]]
    if not training:
        raise ValueError("No verified useful training teachers")
    sources = [Path(__file__), Path(learn.__file__), Path(__file__).with_name("model.py")]
    protocol = dict(data=str(args.data), data_sha256=sha256(args.data.read_bytes()).hexdigest(),
        parent=str(args.checkpoint), parent_sha256=sha256(args.checkpoint.read_bytes()).hexdigest(),
        source_hashes={str(p): sha256(p.read_bytes()).hexdigest() for p in sources},
        states=args.states, train_teacher_states=len(training), epochs=args.epochs, seed=args.seed,
        lr=.0003, arms=["teacher", "repair"], teacher_limit=8, candidate_limit=64, probes=80,
        mutations=0, config=parent.config, attempts=32, total_work=None,
        note="Development diagnostic, not held-out testing. Fixed teacher pools; equal optimizer updates. "
             "Repair uses extra planning calls. Evaluation loss disables exploration. "
             "Departure from measured teachers is not proof of uselessness.")
    write(args.output / "protocol.json", protocol)
    write(args.output / "states.json", records)
    baseline = probe(parent, state, records, args.seed)
    write(args.output / "initial.json", baseline)
    for mode in ("plain", "heuristic", "trained"):
        result = solver_probe(parent, state, validation, mode=mode)
        write(args.output / ("baseline_" + mode + ".json"), result)
        print("baseline", mode, result["nodes"], flush=True)
    checkpoints = {1, 5, 10, 20, args.epochs}
    for arm in ("teacher", "repair"):
        model = deepcopy(parent)
        optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
        rng, history = Random(args.seed), []
        planning = 0
        original = learn.prepare_candidate

        def counted(*a, **k):
            nonlocal planning
            planning += 1
            return original(*a, **k)

        learn.prepare_candidate = counted
        try:
            for epoch in range(1, args.epochs + 1):
                began = perf_counter()
                order = list(training)
                rng.shuffle(order)
                loss_sum = 0.
                model.train()
                for record in order:
                    with _closed(state, record, model.network["colors"]):
                        optimizer.zero_grad()
                        teachers = teachers_for(record)
                        loss = (learn.teacher_loss(model, state.domains, teachers, Random(args.seed + epoch * 1000 + record["id"]))
                                if arm == "teacher" else learn.on_policy_teacher_loss(model, state, teachers))
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                        loss_sum += float(loss.detach())
                row = dict(epoch=epoch, training_loss=loss_sum / len(order), updates=epoch * len(order),
                           seconds=perf_counter() - began, planning_calls=planning)
                if epoch in checkpoints:
                    row["probe"] = probe(model, state, records, args.seed)
                    write(args.output / f"{arm}.epoch{epoch:04d}.json", row)
                history.append(row)
                print(arm, epoch, json.dumps({k: v for k, v in row.items() if k != "probe"}),
                      json.dumps(row.get("probe", {}).get("groups", {})), flush=True)
        finally:
            learn.prepare_candidate = original
        result = solver_probe(model, state, validation, mode="trained")
        write(args.output / (arm + ".dfs.json"), result)
        write(args.output / (arm + ".history.json"), history)
        # NumPy scalars in solver diagnostics must not prevent weights-only loading.
        _save(args.output / (arm + ".pt"), model, json.loads(json.dumps(history)), 0., 1.,
              json.loads(json.dumps(result)),
              training={"method": "fixed_teacher_diagnostic", "objective": "pruning",
                        "parent": dict(path=str(args.checkpoint), sha256=protocol["parent_sha256"]),
                        "train_puzzles": sorted(set(saved["training"].get("train_puzzles", ())) |
                                                {r["puzzle"] for r in training}),
                        "train_instance_hashes": sorted(set(inherited) | learn._instance_hashes(
                            data, [initial[r["puzzle"]] for r in training])),
                        "epoch": args.epochs, "epochs": args.epochs, "planned_epochs": args.epochs,
                        "warmup": args.epochs, "phase": "teacher_only", "teacher_candidates": 64,
                        "teacher_probes": 80, "teacher_mutations": 0, "teacher_limit": 8,
                        "on_policy_imitation": arm == "repair", "imitation_weight": 1.,
                        "seed": args.seed, "lr": .0003, "device": "cpu", "validation_device": "cpu",
                        "cluster_attempts": 32, "max_total_work": None, "validation_seconds": 120.,
                        "development_only": True, "diagnostic": protocol})
        print(arm, "final DFS", result["nodes"], flush=True)
    if sha256(args.data.read_bytes()).hexdigest() != protocol["data_sha256"] or sha256(args.checkpoint.read_bytes()).hexdigest() != protocol["parent_sha256"]:
        raise AssertionError("Source data or checkpoint changed")


if __name__ == "__main__":
    main()
