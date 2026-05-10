import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import pandas as pd
import pyarrow.parquet as pq

from demonstrations.sudoku.examples import standard_sudoku
from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence

from .evaluate import summarize_state
from .run_experiment import solve_task, solve_task_with_backtracking
from .tasks.sudoku import build_constraint_sudoku_task


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "demonstrations" / "sudoku" / "sudoku_bench" / "sample_data"
DEFAULT_DATASET = "hf:train"
REPO_HF_SUDOKU_DATASET_DIR = DATA_DIR / "hf_sudoku_dataset"


def load_sudoku_dataframe(path: Path):
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset file type: {path}")


def hf_sudoku_shards(split: str = "train"):
    shards = []
    for shard_path in REPO_HF_SUDOKU_DATASET_DIR.glob(f"{split}_*.parquet"):
        shard_index = int(shard_path.stem.removeprefix(f"{split}_"))
        shards.append((shard_index, shard_path))
    if not shards:
        raise FileNotFoundError(
            f"Could not find repo-local Sudoku dataset shards for split {split!r} under {REPO_HF_SUDOKU_DATASET_DIR}."
        )
    return [path for _, path in sorted(shards)]


def iter_dataset_rows(dataset, limit: int | None, offset: int, batch_size: int = 256):
    if isinstance(dataset, str) and dataset.startswith("hf:"):
        split = dataset.split(":", 1)[1] or "train"
        seen = 0
        emitted = 0
        for shard_path in hf_sudoku_shards(split):
            parquet_file = pq.ParquetFile(shard_path)
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                rows = batch.to_pylist()
                for row in rows:
                    if seen < offset:
                        seen += 1
                        continue
                    if limit is not None and emitted >= limit:
                        return
                    yield seen, row
                    seen += 1
                    emitted += 1
        return

    df = load_sudoku_dataframe(Path(dataset))
    if limit is not None:
        df = df.iloc[offset:offset + limit]
    else:
        df = df.iloc[offset:]
    for index, row in df.iterrows():
        yield index, row


def build_dataset_constraint_task(row, index: int, num: int = 3):
    puzzle = row["initial_board"] if "initial_board" in row else row["puzzle"]
    solution = row["solution"]
    initial_evidence = initial_board_into_evidence(str(puzzle))
    solution_evidence = initial_board_into_evidence(str(row["solution"]))
    return build_constraint_sudoku_task(
        network_factory=lambda evidence: standard_sudoku.get_assignment_as_constraint_network(
            num=num,
            startAssignment=evidence,
        ),
        initial_evidence=initial_evidence,
        solution_evidence=solution_evidence,
        num=num,
        name=f"dataset_constraint_sudoku_{index}",
    )


def run_dataset(
    dataset_path=DEFAULT_DATASET,
    limit: int | None = None,
    offset: int = 0,
    max_steps: int = 0,
    simulations: int = 0,
    backtracking: bool = False,
):
    solve = solve_task_with_backtracking if backtracking else solve_task
    rows = []
    start_time = perf_counter()
    for index, row in iter_dataset_rows(dataset_path, limit=limit, offset=offset):
        task = build_dataset_constraint_task(row, index=index)
        t0 = perf_counter()
        state, history, cache_info = solve(task, max_steps=max_steps, simulations=simulations)
        elapsed = perf_counter() - t0
        result = asdict(summarize_state(str(index), task, state, history, cache_info))
        result.update(
            {
                "index": int(index),
                "puzzle_id": row.get("puzzle_id", index),
                "elapsed_seconds": elapsed,
            }
        )
        rows.append(result)
        print(
            f"{index}: solved={result['solved']} contradiction={result['contradiction']} "
            f"assigned={result['assigned_count']}/{result['target_count']} "
            f"exact={result['exact_match']} steps={result['history_length']} "
            f"time={elapsed:.3f}s"
        )

    total = len(rows)
    solved = sum(row["solved"] for row in rows)
    exact = sum(row["exact_match"] for row in rows)
    contradictions = sum(row["contradiction"] for row in rows)
    total_elapsed = sum(row["elapsed_seconds"] for row in rows)
    avg_elapsed = total_elapsed / total if total else 0.0
    solved_elapsed = sum(row["elapsed_seconds"] for row in rows if row["solved"])
    avg_solved_elapsed = solved_elapsed / solved if solved else 0.0
    print(
        f"Summary: total={total} solved={solved} exact={exact} "
        f"contradictions={contradictions} avg_time={avg_elapsed:.3f}s "
        f"avg_solved_time={avg_solved_elapsed:.3f}s "
        f"wall_time={perf_counter() - start_time:.3f}s"
    )
    return rows


def write_csv(rows, output_path: Path):
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run AlphaReason constraint-network Sudoku dataset experiments.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Dataset path, or hf:train / hf:valid for the cached HuggingFace Ritvik19/Sudoku-Dataset.",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--simulations", type=int, default=0)
    parser.add_argument("--backtracking", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = run_dataset(
        dataset_path=args.dataset,
        limit=args.limit,
        offset=args.offset,
        max_steps=args.max_steps,
        simulations=args.simulations,
        backtracking=args.backtracking,
    )
    if args.output:
        write_csv(rows, args.output)


if __name__ == "__main__":
    main()
