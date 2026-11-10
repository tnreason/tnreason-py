"""Stream Sudoku shards through the solver's current GAC on one fixed network."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from itertools import islice
import json
import math
from multiprocessing import get_context
from numbers import Integral
from pathlib import Path
from random import Random
import signal
from time import perf_counter

import pyarrow.parquet as pq


_STATE = None
_UNIQUENESS = None


def normalize(board):
    board = str(board).replace("0", ".")
    if len(board) != 81 or set(board) - set(".123456789"):
        raise ValueError("Expected an 81-character Sudoku grid")
    return board


def _valid_solution(puzzle, solution):
    if not isinstance(solution, str) or len(solution) != 81 or set(solution) != set("123456789"):
        return False
    if any(given != "." and given != value for given, value in zip(puzzle, solution)):
        return False
    groups = ([solution[9 * row + col] for col in range(9)] for row in range(9))
    columns = ([solution[9 * row + col] for row in range(9)] for col in range(9))
    boxes = ([solution[9 * (row + dr) + col + dc] for dr in range(3) for dc in range(3)]
             for row in (0, 3, 6) for col in (0, 3, 6))
    return all(len(set(group)) == 9 for groupset in (groups, columns, boxes) for group in groupset)


def check_uniqueness(network, domains, reference, *, nodes=10_000, seconds=10.):
    """Certify uniqueness on selected colors using a linear tensor blocking chain.

    The reference must extend to a solution of the original cores. A second
    solution must differ on at least one reference color; auxiliary assignments
    do not count. Only exhaustive UNSAT of that blocked problem proves unique.
    Each reference/alternative solve has its own node and wall-time guard.
    """
    import numpy as np
    from tnreason.engine.workload_to_numpy import NumpyCore
    from experiments.constraint_networks.constraint_networks import ConstraintNetwork

    if not isinstance(nodes, int) or isinstance(nodes, bool) or nodes < 1 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Expected positive uniqueness node and time limits")
    base = ConstraintNetwork(dict(network.coresDict), maintain_indices=True)
    if not reference or set(reference) - base.dimension_dict.keys():
        raise ValueError("Reference must select at least one existing color")
    if any(not isinstance(value, Integral) or isinstance(value, bool) or not 0 <= value < base.dimension_dict[color]
           for color, value in reference.items()):
        raise ValueError("Reference values must be valid domain indices")

    def valid(assignment):
        return (set(base.dimension_dict) <= assignment.keys()
                and all(assignment[color] in values for color, values in domains.items())
                and all(core[{color: assignment[color] for color in core.colors}] != 0
                        for core in base.coresDict.values()))

    def run(problem, initial):
        def timeout(*_):
            raise TimeoutError("Uniqueness solve exceeded its wall limit")

        started = perf_counter()
        previous = signal.signal(signal.SIGALRM, timeout)
        timer = signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            result = problem.solve_mac(domains=initial, max_nodes=nodes)
            status = ("solved" if result.solved else "unsat" if result.unsatisfiable else "node_limit")
            return status, result
        except TimeoutError:
            return "wall_timeout", None
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            if timer[0]:
                signal.setitimer(signal.ITIMER_REAL, max(1e-9, timer[0] - (perf_counter() - started)), timer[1])

    started = perf_counter()
    reference_domains = dict(domains)
    if any(color in domains and value not in domains[color] for color, value in reference.items()):
        return dict(status="invalid_reference", reference_status="conflicts_with_domains", seconds=0.)
    reference_domains.update({color: (int(value),) for color, value in reference.items()})
    status, result = run(base, reference_domains)
    info = dict(reference_status=status, reference_nodes=0 if result is None else result.nodes,
                reference_colors=len(reference))
    if status != "solved":
        return dict(info, status="invalid_reference" if status == "unsat" else "unknown",
                    seconds=perf_counter() - started)
    if not valid(result.assignment) or any(result.assignment[c] != v for c, v in reference.items()):
        raise AssertionError("Reference solve returned an invalid original-network assignment")

    # q[i+1] = q[i] OR (x[i] != reference[i]); q[0]=False, q[n]=True.
    # Every added tensor has at most three axes and 4*dim(x[i]) entries.
    prefix, count = "__alternative", len(reference)
    used = set(base.dimension_dict) | set(base.coresDict)
    while ({f"{prefix}_start", f"{prefix}_end"} & used or
           any(f"{prefix}_{kind}_{i}" in used for kind in ("state", "step") for i in range(count + 1))):
        prefix += "_"
    flags = [f"{prefix}_state_{i}" for i in range(count + 1)]
    additions = {f"{prefix}_start": NumpyCore(values=np.array([True, False]), colors=[flags[0]]),
                 f"{prefix}_end": NumpyCore(values=np.array([False, True]), colors=[flags[-1]])}
    for idx, (color, value) in enumerate(reference.items()):
        before, current, after = np.indices((2, base.dimension_dict[color], 2))
        additions[f"{prefix}_step_{idx}"] = NumpyCore(
            values=after == ((before == 1) | (current != value)),
            colors=[flags[idx], color, flags[idx + 1]])
    problem = ConstraintNetwork({**base.coresDict, **additions}, maintain_indices=True)
    status, result = run(problem, domains)
    info.update(alternative_status=status, alternative_nodes=0 if result is None else result.nodes)
    if status == "solved":
        if not valid(result.assignment) or not any(result.assignment[c] != v for c, v in reference.items()):
            raise AssertionError("Blocked solve returned an invalid alternative assignment")
        info["alternative"] = {color: result.assignment[color] for color in reference}
    return dict(info, status="multiple" if status == "solved" else "unique" if status == "unsat" else "unknown",
                seconds=perf_counter() - started)


def _initialize(data, unique_only=False, uniqueness_nodes=10_000, uniqueness_seconds=10.):
    global _STATE, _UNIQUENESS
    from .train import _state
    raw = data["network"]
    _STATE = _state(dict(version=1, network=raw, reward_colors=data["reward_colors"],
                         samples=[dict(domains=[list(range(dim)) for dim in raw["dimensions"]])]))
    _UNIQUENESS = dict(nodes=uniqueness_nodes, seconds=uniqueness_seconds) if unique_only else None


def _gac_status(board):
    from ..supervised.collect import sudoku_domains
    _STATE.push_level()
    try:
        for color, values in sudoku_domains(board).items():
            _STATE.restrict_domain(color, values)
        return ("contradiction" if not _STATE.propagate() else
                "solved" if _STATE.is_solved() else "unresolved")
    finally:
        _STATE.pop_level()


def _check(rows):
    """Reuse one unconditioned tensor network and undo every puzzle's givens."""
    results = []
    for row in rows:
        if not _valid_solution(row["puzzle"], row["solution"]):
            results.append(("invalid_solution", row))
            continue
        status = _gac_status(row["puzzle"])
        if status == "contradiction":
            raise AssertionError("Current GAC contradicted a verified Sudoku solution")
        if status == "unresolved" and _UNIQUENESS is not None:
            from ..supervised.collect import sudoku_domains
            reference = {color: values[0] for color, values in sudoku_domains(row["solution"]).items()}
            checked = check_uniqueness(_STATE.constraint_network, sudoku_domains(row["puzzle"]),
                                       reference, **_UNIQUENESS)
            status = "invalid_solution" if checked["status"] == "invalid_reference" else checked["status"]
            row = dict(row, uniqueness=checked)
        results.append((status, row))
    return results


def source_files(directory, split="train"):
    """Prioritize higher source difficulty, then shuffle row groups within shards.

    Source labels only control scan order; inclusion always requires current GAC.
    The first row group's difficulty statistics determine a shard's priority.
    """
    found = []
    for path in sorted(Path(directory).glob(f"{split}_*.parquet")):
        source = pq.ParquetFile(path)
        if not {"puzzle", "solution"} <= set(source.schema.names):
            raise ValueError(f"Missing puzzle/solution columns: {path}")
        difficulty = 0
        if "difficulty" in source.schema.names and source.num_row_groups:
            stats = source.metadata.row_group(0).column(source.schema.names.index("difficulty")).statistics
            if stats is not None and stats.has_min_max:
                difficulty = stats.max
        found.append(dict(path=str(path), rows=source.metadata.num_rows,
                          row_groups=source.num_row_groups, first_group_difficulty=difficulty))
    if not found:
        raise FileNotFoundError(f"No {split}_*.parquet shards in {directory}")
    return sorted(found, key=lambda item: (-item["first_group_difficulty"], item["rows"], item["path"]))


def _rows(sources, seed, batch_size):
    rng = Random(seed)
    for info in sources:
        source = pq.ParquetFile(info["path"])
        groups = list(range(source.num_row_groups))
        rng.shuffle(groups)
        columns = [key for key in ("puzzle", "solution", "difficulty", "missing", "set")
                   if key in source.schema.names]
        for group in groups:
            offset = 0
            for batch in source.iter_batches(batch_size=batch_size, row_groups=[group], columns=columns):
                for row in batch.to_pylist():
                    yield {**row, "puzzle": normalize(row["puzzle"]),
                           "source": dict(path=info["path"], row_group=group, offset=offset)}
                    offset += 1


def reserve_validation(directory, batch_size=1024):
    """Reserve exact puzzles in the source's validation shards without solving them."""
    reserved = set()
    for info in source_files(directory, "valid"):
        for batch in pq.ParquetFile(info["path"]).iter_batches(batch_size=batch_size, columns=["puzzle"]):
            reserved.update(normalize(board) for board in batch.column(0).to_pylist())
    return reserved


def prepare(data, directory, *, target=10_000, workers=4, seed=0, batch_size=64,
            max_scanned=None, drop_validation=(85, 91), reserved=(), verbose=False,
            unique_only=False, uniqueness_nodes=10_000, uniqueness_seconds=10., shards=()):
    """Append new GAC-stalled training puzzles; retain existing puzzle indices.

    ``target`` counts new puzzles, and ``max_scanned`` counts source rows before
    exclusion/deduplication. At most one worker batch per worker is in flight.
    All puzzles in a completed wave count as checked, including target overshoot.
    Optional uniqueness certification uses bounded plain MAC search with generic
    blocking tensors. The default filter uses GAC only, without search.
    """
    if min(target, workers, batch_size) < 1 or max_scanned is not None and max_scanned < 1:
        raise ValueError("Expected positive target, workers, batch size and optional scan limit")
    if (not isinstance(uniqueness_nodes, int) or isinstance(uniqueness_nodes, bool) or uniqueness_nodes < 1
            or not math.isfinite(uniqueness_seconds) or uniqueness_seconds <= 0):
        raise ValueError("Expected positive uniqueness node and time limits")
    train, validation = list(data["train_indices"]), list(data["validation_indices"])
    if set(train) & set(validation):
        raise ValueError("Training and validation indices overlap")
    dropped = sorted(set(drop_validation) & set(validation))
    train += dropped
    validation = [idx for idx in validation if idx not in dropped]
    puzzles = list(data["puzzles"])
    seen = {normalize(board) for board in puzzles}
    excluded = {normalize(board) for board in reserved}
    excluded.update(normalize(puzzles[idx]) for idx in validation)
    sources = source_files(directory)
    if shards:
        missing = set(shards) - {Path(item["path"]).name for item in sources}
        if missing:
            raise ValueError(f"Unknown training shards: {sorted(missing)}")
        sources = [item for item in sources if Path(item["path"]).name in shards]
    stats = dict(source_rows=0, excluded=0, duplicate=0, checked=0, solved=0,
                 unresolved=0, invalid_solution=0, retained=0)
    if unique_only:
        stats.update(unique=0, multiple=0, unknown=0)
    kept, started = [], perf_counter()

    def batches():
        rows = _rows(sources, seed, batch_size)
        if max_scanned is not None:
            rows = islice(rows, max_scanned)
        pending = []
        for row in rows:
            stats["source_rows"] += 1
            board = row["puzzle"]
            if board in excluded:
                stats["excluded"] += 1
                continue
            if board in seen:
                stats["duplicate"] += 1
                continue
            seen.add(board)
            pending.append(row)
            if len(pending) == batch_size:
                yield pending
                pending = []
        if pending:
            yield pending

    pool = None
    if workers == 1:
        _initialize(data, unique_only, uniqueness_nodes, uniqueness_seconds)
    else:
        pool = ProcessPoolExecutor(max_workers=workers, initializer=_initialize,
                                   initargs=(data, unique_only, uniqueness_nodes, uniqueness_seconds),
                                   mp_context=get_context("spawn"))
    try:
        batches_iter = iter(batches())
        while wave := list(islice(batches_iter, workers)):
            futures = [pool.submit(_check, batch) for batch in wave] if pool else None
            results = [future.result() for future in futures] if pool else [_check(wave[0])]
            for result in results:
                for status, row in result:
                    stats["checked"] += 1
                    stats[status] += 1
                    if unique_only and status in ("unique", "multiple", "unknown"):
                        stats["unresolved"] += 1
                    eligible = status == ("unique" if unique_only else "unresolved")
                    if eligible and len(kept) < target:
                        kept.append(row)
                    elif not eligible:
                        # Retained puzzles need exact deduplication. Forget solved
                        # rows so a scan of a huge easy shard stays bounded.
                        seen.discard(row["puzzle"])
            stats["retained"] = len(kept)
            if verbose:
                print(f"GAC checked {stats['checked']:,}; retained {len(kept):,}/{target:,}; "
                      f"solved {stats['solved']:,}; excluded {stats['excluded']:,}; "
                      f"elapsed {perf_counter() - started:.1f}s", flush=True)
            if len(kept) == target:
                break
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    # Existing training puzzles may already be solved by GAC. Count only these
    # few old roots again, rather than rechecking all newly verified roots.
    if workers != 1:
        _initialize(data)
    existing_unresolved = sum(_gac_status(normalize(puzzles[idx])) == "unresolved" for idx in train)
    first = len(puzzles)
    puzzles.extend(row["puzzle"] for row in kept)
    train.extend(range(first, len(puzzles)))
    metadata = dict(method="current_mac_gac", seed=seed, target=target, workers=workers,
        batch_size=batch_size, max_scanned=max_scanned, source_files=sources,
        source_data_sha256=sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest(),
        elapsed_s=perf_counter() - started, source_validation_reserved=len(excluded),
        dropped_validation_indices=dropped, retained_validation_indices=validation,
        new_train_indices=list(range(first, len(puzzles))), target_reached=len(kept) == target,
        existing_unresolved_train_puzzles=existing_unresolved,
        unresolved_train_puzzles=existing_unresolved + len(kept),
        stats=stats, records=[dict(puzzle_index=first + idx, **row) for idx, row in enumerate(kept)])
    if shards:
        metadata["selected_shards"] = list(shards)
    if unique_only:
        metadata.update(method="current_mac_gac_unique", unique_only=True,
                        uniqueness_nodes=uniqueness_nodes, uniqueness_seconds=uniqueness_seconds,
                        uniqueness_scope="Source solution position colors; generic OR-chain blocking tensors",
                        existing_training_uniqueness_checked=False)
    # Original saved branch states are deliberately omitted: this dataset consists
    # of independent initial puzzles, closed by training_states on loading.
    result = {key: value for key, value in data.items()
              if key not in {"samples", "branch_preparation", "rejected"}}
    return {**result, "puzzles": puzzles, "train_indices": train,
            "validation_indices": validation, "scale_preparation": metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing dataset supplies the unchanged tensor network and splits")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-scanned", type=int)
    parser.add_argument("--shards", nargs="+", default=[], help="Training shard basenames to scan, e.g. train_0.parquet")
    parser.add_argument("--unique-only", action="store_true", help="Retain only certified unique, GAC-stalled puzzles")
    parser.add_argument("--uniqueness-nodes", type=int, default=10_000, help="Node guard per reference/alternative solve")
    parser.add_argument("--uniqueness-seconds", type=float, default=10., help="Wall-time guard per reference/alternative solve")
    parser.add_argument("--drop-validation", type=int, nargs="*", default=[85, 91])
    parser.add_argument("--reserved-data", type=Path,
                        help="Existing held-out dataset; reserve its explicitly reserved indices")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a fresh filename")
    data = json.loads(args.source.read_text())
    reserved = reserve_validation(args.dataset_dir)
    if args.reserved_data is not None:
        heldout = json.loads(args.reserved_data.read_text())
        indices = data.get("development_provenance", {}).get("still_reserved_testing_indices",
                                                           range(len(heldout["puzzles"])))
        reserved.update(heldout["puzzles"][idx] for idx in indices)
    print(f"Reserved {len(reserved):,} exact validation/test puzzles; filtering with current MAC GAC", flush=True)
    result = prepare(data, args.dataset_dir, target=args.target, workers=args.workers, seed=args.seed,
                     batch_size=args.batch_size, max_scanned=args.max_scanned,
                     drop_validation=args.drop_validation, reserved=reserved, verbose=True,
                     unique_only=args.unique_only, uniqueness_nodes=args.uniqueness_nodes,
                     uniqueness_seconds=args.uniqueness_seconds, shards=args.shards)
    result["scale_preparation"].update(source_path=str(args.source),
                                       source_file_sha256=sha256(args.source.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, separators=(",", ":"))
    stats = result["scale_preparation"]["stats"]
    print(f"Wrote {stats['retained']:,} new training puzzles to {args.output}; "
          f"target reached: {result['scale_preparation']['target_reached']}", flush=True)


if __name__ == "__main__":
    main()
