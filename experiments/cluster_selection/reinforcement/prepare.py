"""Add generic MAC branch states from explicit training instances only."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from random import Random

from .experiment import fingerprints, instances
from .train import _state


def prepare(data, *, states_per_puzzle=16, nodes=512, seed=0, max_puzzles=None, verbose=False):
    """Sample bounded random branch walks; each saved state retains its root ID.

    These are additional domain contexts from existing training problems, not
    independent new puzzles. No cluster is evaluated and no validation is run.
    """
    if (any(not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in (states_per_puzzle, nodes))
            or not isinstance(seed, int) or isinstance(seed, bool)
            or max_puzzles is not None and (not isinstance(max_puzzles, int)
                or isinstance(max_puzzles, bool) or max_puzzles < 1)):
        raise ValueError("Expected positive state/node limits and an integer seed")
    initial, _ = instances(data)
    train, validation = data.get("train_indices"), data.get("validation_indices")
    if (not train or validation is None or any(not isinstance(idx, int) or isinstance(idx, bool)
            or idx < 0 or idx >= len(initial) for idx in (*train, *validation))
            or len(set(train)) != len(train) or len(set(validation)) != len(validation)
            or set(train) & set(validation)):
        raise ValueError("Preparation requires valid explicit disjoint training/validation indices")
    if fingerprints(data, [initial[idx] for idx in train]) & fingerprints(
            data, [initial[idx] for idx in validation]):
        raise ValueError("Training and validation initial domains overlap")
    samples = list(data.get("samples", ()))
    if any(sample.get("puzzle") not in train for sample in samples):
        raise ValueError("Saved states must belong to explicit training instances")
    raw = data["network"]
    base = {**data, "reward_colors": data.get("reward_colors", raw["colors"]),
            "samples": samples or [dict(domains=[list(range(dim)) for dim in raw["dimensions"]])]}
    network = _state(base).constraint_network
    colors, rng = raw["colors"], Random(seed)
    seen = {tuple(tuple(values) for values in sample["domains"]) for sample in samples}
    totals = dict(train_instances=len(train), visited_roots=0, solved_roots=0,
                  contradictory_roots=0, root_states=0, new_states=0, nodes=0,
                  duplicates=0, contradictions=0, solved_branches=0, rows=[])
    for puzzle in train[:max_puzzles]:
        state = network.create_mac_state(initial[puzzle])
        totals["visited_roots"] += 1
        if not state.propagate() or state.is_solved():
            totals["contradictory_roots" if state.contradiction else "solved_roots"] += 1
            continue
        before = len(samples)

        def save():
            snapshot = tuple(tuple(state.domains[color]) for color in colors)
            if snapshot in seen:
                totals["duplicates"] += 1
                return
            seen.add(snapshot)
            samples.append(dict(puzzle=puzzle, level=state.decision_level, domains=snapshot,
                                source="training_mac_branch", candidates=[]))
            totals["new_states"] += 1
            totals["root_states"] += int(state.decision_level == 0)

        save()
        used = 0
        try:
            while len(samples) - before < states_per_puzzle and used < nodes:
                # Restart part of a walk to diversify contexts near the root.
                if state.decision_level and rng.random() < .25:
                    for _ in range(rng.randint(1, state.decision_level)):
                        state.pop_level()
                active = [color for color in base["reward_colors"] if len(state.domains[color]) > 1]
                if not active:
                    active = [color for color in colors if len(state.domains[color]) > 1]
                smallest = min(len(state.domains[color]) for color in active)
                color = rng.choice([color for color in active if len(state.domains[color]) == smallest])
                value = rng.choice(state.domains[color])
                state.push_level()
                state.assign(color, value)
                used += 1
                totals["nodes"] += 1
                if state.propagate() and not state.is_solved():
                    save()
                else:
                    totals["contradictions" if state.contradiction else "solved_branches"] += 1
                    while state.decision_level:
                        state.pop_level()
        finally:
            while state.decision_level:
                state.pop_level()
        totals["rows"].append(dict(puzzle=puzzle, new_states=len(samples) - before, nodes=used))
        if verbose:
            print(f"puzzle {puzzle}: {len(samples) - before} new states, {used} branch nodes", flush=True)
    metadata = dict(method="random_training_mac_branches", seed=seed, states_per_puzzle=states_per_puzzle,
                    max_nodes_per_puzzle=nodes, max_puzzles=max_puzzles, restart_probability=.25,
                    source_data_sha256=sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest(),
                    source_sample_count=len(data.get("samples", ())), **totals)
    return {**data, "samples": samples,
            "branch_preparation": [*data.get("branch_preparation", ()), metadata]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states-per-puzzle", type=int, default=16)
    parser.add_argument("--nodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-puzzles", type=int, help="Limit preparation only; preserve all dataset splits")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a fresh path")
    source = args.source.read_bytes()
    result = prepare(json.loads(source), states_per_puzzle=args.states_per_puzzle, nodes=args.nodes,
                     seed=args.seed, max_puzzles=args.max_puzzles, verbose=True)
    metadata = result["branch_preparation"][-1]
    metadata.update(source_path=str(args.source), source_file_sha256=sha256(source).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, separators=(",", ":"))
    print(f"Added {metadata['new_states']} states from {metadata['visited_roots']} training roots "
          f"with {metadata['nodes']} branch nodes; wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
