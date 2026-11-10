"""Stream Sudoku shards through the solver's current GAC on one fixed network."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from itertools import islice
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
from time import perf_counter

import pyarrow.parquet as pq


_STATE = None


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


def _initialize(data):
    global _STATE
    from .train import _state
    raw = data["network"]
    _STATE = _state(dict(version=1, network=raw, reward_colors=data["reward_colors"],
                         samples=[dict(domains=[list(range(dim)) for dim in raw["dimensions"]])]))


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
            max_scanned=None, drop_validation=(85, 91), reserved=(), verbose=False):
    """Append new GAC-stalled training puzzles; retain existing puzzle indices.

    ``target`` counts new puzzles, and ``max_scanned`` counts source rows before
    exclusion/deduplication. At most one worker batch per worker is in flight.
    All puzzles in a completed wave count as checked, including target overshoot.
    No teacher contraction, cluster generation or search is used to filter rows.
    """
    if min(target, workers, batch_size) < 1 or max_scanned is not None and max_scanned < 1:
        raise ValueError("Expected positive target, workers, batch size and optional scan limit")
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
    stats = dict(source_rows=0, excluded=0, duplicate=0, checked=0, solved=0,
                 unresolved=0, invalid_solution=0, retained=0)
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
        _initialize(data)
    else:
        pool = ProcessPoolExecutor(max_workers=workers, initializer=_initialize, initargs=(data,),
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
                    if status == "unresolved" and len(kept) < target:
                        kept.append(row)
                    elif status != "unresolved":
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
                     drop_validation=args.drop_validation, reserved=reserved, verbose=True)
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
