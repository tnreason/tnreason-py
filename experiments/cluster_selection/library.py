"""Build a fixed library from successful applications in training-only heuristic MAC runs.

Run with a dataset containing a raw network and explicit train_indices. One
atomic record is retained per puzzle; --resume reuses completed records.
The final library contains core/output identities, never cached summary values.
Only measured failures are negatives; unrecorded outcomes remain unknown.
Use one collector per output directory, including when resuming. For example:

python -m experiments.cluster_selection.library DATA.json --output NEW_DIRECTORY
    --puzzles 1000 --allowance 256 --workers 4 --record-failures
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from random import Random
import signal
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from .data import network_data
from .runtime import attach_heuristic


def _training(data):
    """Require an explicit split; exclude matching domains in every reserved row."""
    if "initial_domains" in data:
        initial = data["initial_domains"]
    else:
        from .supervised.collect import sudoku_domains
        initial = [sudoku_domains(board, round(len(board) ** .25)) for board in data["puzzles"]]
    train = data.get("train_indices")
    reserved = [*data.get("validation_indices", ()), *data.get("test_indices", ())]
    if (not train or any(type(i) is not int or not 0 <= i < len(initial) for i in (*train, *reserved))
            or len(set(train)) != len(train) or set(train) & set(reserved)):
        raise ValueError("Library collection requires explicit, valid, disjoint training indices")
    dimensions = dict(zip(data["network"]["colors"], data["network"]["dimensions"]))

    def fingerprint(domains):
        if set(domains) - dimensions.keys():
            raise ValueError("Unknown initial-domain variables")
        restricted = []
        for color, values in domains.items():
            if any(type(v) is not int or not 0 <= v < dimensions[color] for v in values):
                raise ValueError("Invalid initial-domain values")
            values = tuple(sorted(set(values)))
            if values != tuple(range(dimensions[color])):
                restricted.append((color, values))
        return tuple(sorted(restricted))

    train_set = set(train)
    heldout = {fingerprint(domains) for i, domains in enumerate(initial) if i not in train_set}
    if heldout & {fingerprint(initial[i]) for i in train}:
        raise ValueError("Training and reserved initial domains overlap")
    return initial, list(train)


def _write(path, value):
    """Publish a complete new record; a killed puzzle can be retried on resume."""
    if path.exists():
        raise FileExistsError(path)
    with NamedTemporaryFile(mode="w", dir=path.parent, prefix=".library-", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, separators=(",", ":"))
    try:
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)


def _run(network, domains, puzzle, reward_colors, config):
    state = network.create_mac_state(domains)
    colors, core_keys = list(network.dimension_dict), list(network.coresDict)
    color_idx, core_idx = {c: i for i, c in enumerate(colors)}, {k: i for i, k in enumerate(core_keys)}
    snapshots, applications, seen = [], [], {}

    def observe(current, candidate, estimate, before, result):
        useful = bool(result.removed_values or result.contradiction)
        if not useful and not config["record_failures"]:
            return
        masks = tuple(sum(1 << value for value in before[color]) for color in colors)
        if masks not in seen:
            seen[masks] = len(snapshots)
            snapshots.append(masks)
        applications.append(dict(state=seen[masks], level=current.decision_level,
            cores=sorted(core_idx[key] for key in candidate.core_keys),
            open=sorted(color_idx[color] for color in candidate.open_colors),
            removed_values=result.removed_values,
            primary_removed_values=sum(len(before[c]) - len(current.domains[c]) for c in reward_colors),
            pruned_values=[dict(variable=color_idx[color], removed=sorted(set(values) - set(current.domains[color])))
                           for color, values in before.items() if values != current.domains[color]],
            contradiction=result.contradiction, work=estimate.work, max_size=estimate.max_size))

    options = {key: config[key] for key in ("max_cores", "max_open_colors", "max_steps",
               "max_work", "max_size", "compact_domains")}
    attach_heuristic(state, max_candidates=config["allowance"], on_application=observe, **options)
    state_policy = state.on_closure

    def timeout(*_):
        raise TimeoutError("Heuristic collection puzzle deadline")

    previous = signal.signal(signal.SIGALRM, timeout)
    started = perf_counter()
    try:
        signal.setitimer(signal.ITIMER_REAL, config["seconds"])
        result = network.solve_mac(state=state, max_nodes=config["nodes"])
        signal.setitimer(signal.ITIMER_REAL, 0)
        valid = result.solved and (all(result.assignment[c] in values for c, values in domains.items())
            and all(core[{c: result.assignment[c] for c in core.colors}] != 0
                    for core in network.coresDict.values()))
        if result.solved and not valid:
            raise AssertionError("Heuristic collection produced an invalid solution")
        status = "solved" if result.solved else "unsat" if result.unsatisfiable else "node_limit"
        nodes = result.nodes
    except TimeoutError:
        status, nodes = "time_limit", None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        state.on_closure = None
    return dict(puzzle=puzzle, status=status, nodes=nodes, seconds=perf_counter() - started,
                statistics=dict(state_policy.statistics), states=snapshots, applications=applications)


def collect(data, output, *, puzzles=None, offset=0, seed=0, nodes=256, seconds=30., allowance=64,
            max_cores=64, max_open_colors=6, max_steps=None, max_work=1_000_000,
            max_size=100_000, compact_domains=False, record_failures=False, workers=1,
            resume=False, verbose=False):
    """Retain every observed useful (cores, outputs) pair, without truncating the library.

    Runs use the existing sequential heuristic, including its updated-domain GAC
    after each application. A time/node-limited run still supplies valid successes.
    Each saved state's masks encode allowed value v as bit (1 << v).
    """
    if (any(type(v) is not int or v < 1 for v in
            (nodes, allowance, max_cores, max_open_colors, max_work, max_size, workers))
            or type(offset) is not int or offset < 0 or type(seed) is not int
            or puzzles is not None and (type(puzzles) is not int or puzzles < 1)
            or not isfinite(seconds) or seconds <= 0
            or max_steps is not None and (type(max_steps) is not int or max_steps < 1)):
        raise ValueError("Expected positive collection budgets and a nonnegative offset")
    initial, train = _training(data)
    Random(seed).shuffle(train)
    train = train[offset:None if puzzles is None else offset + puzzles]
    if not train:
        raise ValueError("No training instances selected")
    raw = data["network"]
    network = ConstraintNetwork({c["key"]: NumpyCore(
        values=np.asarray(c["values"], dtype=bool).reshape(c["shape"]),
        colors=[raw["colors"][i] for i in c["colors"]]) for c in raw["cores"]}, maintain_indices=True)
    reward_colors = data.get("reward_colors", raw["colors"])
    if network_data(network) != raw or set(reward_colors) - set(raw["colors"]):
        raise ValueError("Invalid network schema or reward colors")
    config = dict(seed=seed, offset=offset, nodes=nodes, seconds=seconds, allowance=allowance,
        max_cores=max_cores, max_open_colors=max_open_colors,
        max_steps=max(8192, 256 * allowance) if max_steps is None else max_steps,
        max_work=max_work, max_size=max_size, compact_domains=compact_domains,
        record_failures=record_failures, workers=workers)
    code_files = (Path(__file__), Path(__file__).with_name("runtime.py"),
                  Path(__file__).with_name("candidate_generation.py"),
                  Path(__file__).with_name("candidate_evaluation_mac.py"),
                  Path(__file__).with_name("contraction_cost.py"),
                  Path(__file__).parent.parent / "constraint_networks" / "mac.py",
                  Path(__file__).parent.parent / "constraint_networks" / "constraint_networks.py",
                  Path(__file__).parent.parent / "constraint_networks" / "generalized_arc_consistency.py")
    protocol = dict(version=1, kind="heuristic_cluster_library", network=raw,
        reward_colors=reward_colors, train_indices=train, config=config,
        source_sha256=sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest(),
        code_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in code_files})
    output = Path(output)
    if resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume requires identical data, selected training rows, code and budgets")
        if (output / "library.json").exists():
            raise ValueError("Library is already complete; choose a new output directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "puzzles").mkdir()
        _write(output / "protocol.json", protocol)
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        futures = {puzzle: executor.submit(_run, network, initial[puzzle], puzzle, reward_colors, config)
                   for puzzle in train if not (output / "puzzles" / f"{puzzle:06d}.json").exists()} if executor else {}
        for puzzle in train:
            path = output / "puzzles" / f"{puzzle:06d}.json"
            if not path.exists():
                row = (futures[puzzle].result() if executor else
                       _run(network, initial[puzzle], puzzle, reward_colors, config))
                _write(path, row)
                if verbose:
                    print(f"recorded train puzzle {puzzle}: {row['status']}, "
                          f"{row['statistics']['useful']}/{row['statistics']['applied']} useful", flush=True)
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    clusters, growth, statuses = {}, [], Counter()
    total_trials = total_useful = root_successes = 0
    for number, puzzle in enumerate(train, 1):
        path = output / "puzzles" / f"{puzzle:06d}.json"
        row = json.loads(path.read_text())
        if row["puzzle"] != puzzle:
            raise ValueError("Puzzle record does not match selected training instance")
        statuses[row["status"]] += 1
        total_trials += row["statistics"]["applied"]
        total_useful += row["statistics"]["useful"]
        for trial in row["applications"]:
            if not (trial["removed_values"] or trial["contradiction"]):
                continue
            signature = tuple(trial["cores"]), tuple(trial["open"])
            entry = clusters.setdefault(signature, dict(cores=trial["cores"], open=trial["open"],
                successes=0, root_successes=0, dfs_successes=0, contradictions=0, puzzles=set(),
                max_removed_values=0, max_primary_removed_values=0))
            entry["puzzles"].add(puzzle)
            entry["successes"] += 1
            entry["root_successes" if trial["level"] == 0 else "dfs_successes"] += 1
            root_successes += int(trial["level"] == 0)
            entry["contradictions"] += int(trial["contradiction"])
            for key in ("removed_values", "primary_removed_values"):
                entry["max_" + key] = max(entry["max_" + key], trial[key])
        growth.append(dict(puzzles=number, clusters=len(clusters), useful_applications=total_useful))
        if verbose:
            print(f"puzzle {number}/{len(train)} (train {puzzle}): {row['status']}, "
                  f"{row['statistics']['useful']}/{row['statistics']['applied']} useful, "
                  f"{len(clusters)} distinct clusters", flush=True)
    entries = [{**entry, "puzzles": sorted(entry["puzzles"])} for entry in clusters.values()]
    entries.sort(key=lambda e: (-len(e["puzzles"]), -e["successes"], e["cores"], e["open"]))
    recorded_useful = sum(entry["successes"] for entry in entries)
    library = {**protocol, "clusters": entries, "growth": growth,
        "summary": dict(puzzles=len(train), statuses=dict(statuses), applied=total_trials,
                        useful=total_useful, recorded_useful=recorded_useful, root_successes=root_successes,
                        dfs_successes=recorded_useful - root_successes, distinct_clusters=len(entries)),
        "note": "Training-only observed successes, not a coverage claim. Outcomes in other domains are unknown. "
                "Only identities are frozen; contract with current domains. Collection timings include recording."}
    _write(output / "library.json", library)
    return library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New directory, or an incomplete --resume run")
    parser.add_argument("--puzzles", type=int, help="Limit training instances after seeded shuffling")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--allowance", type=int, default=64)
    parser.add_argument("--max-cores", type=int, default=64)
    parser.add_argument("--max-open-colors", type=int, default=6)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-work", type=int, default=1_000_000)
    parser.add_argument("--max-size", type=int, default=100_000)
    parser.add_argument("--compact-domains", action="store_true")
    parser.add_argument("--record-failures", action="store_true", help="Also retain measured no-op labels")
    parser.add_argument("--workers", type=int, default=1, help="Independent CPU solver processes")
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    data = json.loads(args.pop("data").read_text())
    result = collect(data, **args, verbose=True)
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
