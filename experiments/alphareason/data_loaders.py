import csv
from pathlib import Path
import re
from typing import List, Optional

import pyarrow.parquet as pq

from .task import AlphaReasonTask
from .tasks.sudoku import build_constraint_sudoku_task
from demonstrations.sudoku.examples import standard_sudoku
from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence


REPO_HF_SUDOKU_DATASET_DIR = (
    Path(__file__).resolve().parents[2]
    / "demonstrations"
    / "sudoku"
    / "sudoku_bench"
    / "sample_data"
    / "hf_sudoku_dataset"
)

REPO_CHALLENGE100_CSV_PATH = (
    Path(__file__).resolve().parents[2]
    / "demonstrations"
    / "sudoku"
    / "sudoku_bench"
    / "sample_data"
    / "challenge100.csv"
)


def _require_dir(path: Path, *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")


def _require_file(path: Path, *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")


def _resolve_dataset_file(dataset_dir: Path, file_name: str) -> Path:
    candidate = dataset_dir / file_name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not resolve dataset file {candidate}.")


def _build_constraint_sudoku_task_from_boards(
    *,
    initial_board: str,
    solution_board: str,
    num: int,
    name: str,
    metadata: dict,
) -> AlphaReasonTask:
    initial_evidence = initial_board_into_evidence(str(initial_board), colNum=num, rowNum=num)
    solution_evidence = initial_board_into_evidence(str(solution_board), colNum=num, rowNum=num)
    task = build_constraint_sudoku_task(
        network_factory=lambda evidence: standard_sudoku.get_assignment_as_constraint_network(
            num=num,
            startAssignment=evidence,
        ),
        initial_evidence=initial_evidence,
        solution_evidence=solution_evidence,
        num=num,
        name=name,
    )
    task.metadata.update(metadata)
    return task


def load_hf_sudoku_dataset_training_tasks(
    num_samples: Optional[int] = None,
    split: str = "train",
    batch_size: int = 256,
    num: int = 3,
    dataset_dir: str = str(REPO_HF_SUDOKU_DATASET_DIR),
) -> List[AlphaReasonTask]:
    if num_samples is not None and num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    dataset_path = Path(dataset_dir)
    _require_dir(dataset_path, label="Dataset directory")

    shard_files: List[tuple[int, Path]] = []
    pattern = re.compile(rf"{re.escape(split)}_(\d+)\.parquet$")
    for path in sorted(dataset_path.iterdir(), key=lambda item: item.name):
        match = pattern.fullmatch(path.name)
        if match:
            shard_files.append((int(match.group(1)), _resolve_dataset_file(dataset_path, path.name)))
    if not shard_files:
        raise FileNotFoundError(f"Could not find parquet shards for split {split!r} under {dataset_path}.")

    tasks: List[AlphaReasonTask] = []
    seen = 0
    for shard_idx, parquet_path in shard_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["puzzle", "solution"]):
            batch_dict = batch.to_pydict()
            puzzles = batch_dict["puzzle"]
            solutions = batch_dict["solution"]
            for puzzle, solution in zip(puzzles, solutions):
                task = _build_constraint_sudoku_task_from_boards(
                    initial_board=puzzle,
                    solution_board=solution,
                    num=num,
                    name=f"sudoku_dataset_{split}_{seen}",
                    metadata={
                        "source": "hf_sudoku_dataset",
                        "split": split,
                        "row_index": seen,
                        "shard_file": parquet_path.name,
                        "shard_index": shard_idx,
                    },
                )
                tasks.append(task)
                seen += 1
                if num_samples is not None and len(tasks) >= num_samples:
                    return tasks
    if not tasks:
        raise ValueError("No HF Sudoku training tasks were loaded.")
    return tasks


def load_challenge100_training_tasks(
    num_samples: int = 20,
    *,
    csv_path: str = str(REPO_CHALLENGE100_CSV_PATH),
    num: int = 3,
) -> List[AlphaReasonTask]:
    path = Path(csv_path)
    _require_file(path, label="Challenge100 CSV")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    tasks: List[AlphaReasonTask] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            if len(tasks) >= num_samples:
                break
            initial_board = row.get("initial_board")
            solution = row.get("solution")
            if initial_board is None or solution is None:
                raise ValueError(f"Missing required columns in {path}: got keys={list(row.keys())}")
            puzzle_id = row.get("puzzle_id", row_idx)
            task = _build_constraint_sudoku_task_from_boards(
                initial_board=str(initial_board),
                solution_board=str(solution),
                num=num,
                name=f"challenge100_{puzzle_id}",
                metadata={
                    "source": "challenge100",
                    "row_index": int(row_idx),
                    "puzzle_id": puzzle_id,
                    "title": row.get("title"),
                    "author": row.get("author"),
                },
            )
            tasks.append(task)

    if not tasks:
        raise ValueError(f"No tasks were loaded from {path}.")
    return tasks


def load_prepared_sudoku_training_tasks(
    *,
    parquet_path: str,
    num_samples: Optional[int] = None,
    batch_size: int = 256,
    num: int = 3,
) -> List[AlphaReasonTask]:
    path = Path(parquet_path)
    _require_file(path, label="Prepared parquet")
    if num_samples is not None and num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    parquet_file = pq.ParquetFile(path)
    tasks: List[AlphaReasonTask] = []
    seen = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["puzzle", "solution", "source", "puzzle_id", "name"]):
        b = batch.to_pydict()
        for puzzle, solution, source, puzzle_id, name in zip(
            b["puzzle"],
            b["solution"],
            b["source"],
            b["puzzle_id"],
            b["name"],
        ):
            task = _build_constraint_sudoku_task_from_boards(
                initial_board=puzzle,
                solution_board=solution,
                num=num,
                name=str(name),
                metadata={"source": str(source), "puzzle_id": str(puzzle_id), "prepared": True},
            )
            tasks.append(task)
            seen += 1
            if num_samples is not None and seen >= num_samples:
                return tasks

    if not tasks:
        raise ValueError(f"No tasks were loaded from {path}.")
    return tasks
