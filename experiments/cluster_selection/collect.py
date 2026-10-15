"""Collect independent cluster trials; run with a text or Nikoli parquet puzzle file."""
import argparse
import json
from hashlib import blake2b
from pathlib import Path
import numpy as np
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.sudoku_tests.standard_constraints import get_sudoku_constraint_network
from .candidate_generation import ClusterCandidateGenerator
from .candidate_evaluation_mac import evaluate_candidate


def network_data(network):
    """Raw ordered tensor data, not neural features; store once per dataset."""
    colors = list(network.dimension_dict)
    idx = {color: i for i, color in enumerate(colors)}
    cores = []
    for key, core in network.coresDict.items():
        values = getattr(core, "values", None)
        support = values != 0 if isinstance(values, np.ndarray) else np.array([
            core[dict(zip(core.colors, coordinate))] != 0 for coordinate in np.ndindex(tuple(core.shape))
        ]).reshape(core.shape)
        cores.append(dict(key=key, colors=[idx[color] for color in core.colors],
                          shape=[int(dim) for dim in core.shape], values=support.astype(bool).ravel().tolist()))
    return dict(colors=colors, dimensions=[int(network.dimension_dict[color]) for color in colors], cores=cores)


def collect(network, initial_domains, *, reward_colors=None, max_states=100, states_per_puzzle=8,
            max_nodes=1000, max_candidates=32, max_work=1_000_000, max_size=100_000):
    """Observe plain search without committing trials. Time labels include rollback.

    Domains are stored once per state, candidates alongside it. Limits bound
    sampling/search and each contraction, not total wall time or peak RAM.
    """
    if min(max_states, states_per_puzzle, max_nodes, max_candidates, max_work, max_size) < 1:
        raise ValueError("Collection limits must be positive")
    generator = ClusterCandidateGenerator(network, max_candidates=max_candidates, max_work=max_work, max_size=max_size)
    raw = network_data(ConstraintNetwork(network.coresDict, maintain_indices=True))
    colors = raw["colors"]
    color_idx = {color: i for i, color in enumerate(colors)}
    core_idx = {core["key"]: i for i, core in enumerate(raw["cores"])}
    targets = tuple(dict.fromkeys(colors if reward_colors is None else reward_colors))
    if set(targets) - color_idx.keys():
        raise KeyError("Unknown reward colors")
    samples, seen, rejected = [], set(), 0

    class Full(Exception):
        pass

    for puzzle_idx, domains in enumerate(initial_domains):
        count = 0

        def observe(state, changed):
            nonlocal count, rejected
            if state.is_solved():
                return
            snapshot = tuple(state.domains[color] for color in colors)
            signature = blake2b(repr(snapshot).encode(), digest_size=16).digest()
            if signature in seen:
                return
            seen.add(signature)
            trials = []
            for candidate in generator.generate(state.domains, changed_colors=changed):
                try:
                    result = evaluate_candidate(state, candidate, reward_colors=targets,
                                                max_work=max_work, max_size=max_size)
                except ValueError as error:
                    if str(error) != "Candidate exceeds the contraction work or tensor-size budget":
                        raise
                    rejected += 1
                    continue
                estimate = result.estimate
                trials.append(dict(cores=[core_idx[key] for key in candidate.core_keys],
                                   open=[color_idx[color] for color in candidate.open_colors],
                                   estimate=[estimate.work, estimate.max_size, estimate.output_size],
                                   gain=result.removed_domain_values, time=result.cost,
                                   contradiction=result.contradiction))
            if trials:
                samples.append(dict(puzzle=puzzle_idx, level=state.decision_level,
                                    domains=snapshot, candidates=trials))
                count += 1
                if len(samples) >= max_states or count >= states_per_puzzle:
                    raise Full

        state = network.create_mac_state(domains)
        state.on_closure = observe
        try:
            network.solve_mac(state=state, max_nodes=max_nodes)
        except Full:
            pass
        finally:
            state.on_closure = None
        if len(samples) >= max_states:
            break
    return dict(version=1, network=raw, reward_colors=list(targets), samples=samples, rejected=rejected)


def sudoku_domains(board, num=3):
    """Digit strings use 1..n; 0 and . mean blank. Givens do not add cores."""
    size = num ** 2
    if num not in (1, 2, 3) or len(board) != size ** 2 or set(board) - set("0." + "123456789"[:size]):
        raise ValueError("Expected a 1x1, 4x4 or 9x9 digit grid with 0/. for blanks")
    domains = {}
    for idx, value in enumerate(board):
        if value in "0.":
            continue
        row, col = divmod(idx, size)
        domains[f"pos_{row // num}_{row % num}_{col // num}_{col % num}"] = (int(value) - 1,)
    return domains


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("puzzles", type=Path, help="One grid per line, or parquet with an initial_board column")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--per-puzzle", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--nodes", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new filename")
    if args.puzzles.suffix == ".parquet":
        import pandas as pd
        boards = pd.read_parquet(args.puzzles)["initial_board"].tolist()
    else:
        boards = [line.strip() for line in args.puzzles.read_text().splitlines() if line.strip()]
    domains = [sudoku_domains(board, args.num) for board in boards]
    network = ConstraintNetwork(get_sudoku_constraint_network(args.num), maintain_indices=True)
    print(f"Collecting from {len(boards)} puzzles (up to {args.states} states)...", flush=True)
    data = collect(network, domains, reward_colors=[color for color in network.dimension_dict if color.startswith("pos_")],
                   max_states=args.states, states_per_puzzle=args.per_puzzle,
                   max_candidates=args.candidates, max_nodes=args.nodes)
    if not data["samples"]:
        parser.error("No candidate trials collected; try harder puzzles or larger search/budget limits")
    data["puzzles"] = boards
    with args.output.open("x") as stream:
        json.dump(data, stream, separators=(",", ":"))
    trials = [trial for sample in data["samples"] for trial in sample["candidates"]]
    useful = sum(trial["gain"] > 0 or trial["contradiction"] for trial in trials)
    print(f"{len(data['samples'])} states, {len(trials)} trials, {useful} useful, {data['rejected']} over budget; wrote {args.output}")


if __name__ == "__main__":
    main()
