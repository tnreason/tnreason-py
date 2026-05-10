import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Iterator, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from demonstrations.sudoku.examples import standard_sudoku
from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence
from .closure_engine import AlphaReasonClosureEngine
from .tasks.sudoku import build_constraint_sudoku_task


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHALLENGE100_CSV = REPO_ROOT / "demonstrations" / "sudoku" / "sudoku_bench" / "sample_data" / "challenge100.csv"
DEFAULT_HF_DATASET_DIR = REPO_ROOT / "demonstrations" / "sudoku" / "sudoku_bench" / "sample_data" / "hf_sudoku_dataset"
DEFAULT_OUT_PARQUET = REPO_ROOT / "demonstrations" / "sudoku" / "sudoku_bench" / "sample_data" / "alphareason_prepared_unsolved.parquet"


@dataclass(frozen=True)
class PuzzleRow:
    puzzle: str
    solution: str
    source: str
    name: str
    puzzle_id: str


def _iter_challenge100_rows(path: Path, *, limit: Optional[int]) -> Iterator[PuzzleRow]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            if limit is not None and row_idx >= limit:
                return
            puzzle = row.get("initial_board")
            solution = row.get("solution")
            if puzzle is None or solution is None:
                raise ValueError(f"Missing required columns in {path}: got keys={list(row.keys())}")
            puzzle_id = str(row.get("puzzle_id", row_idx))
            yield PuzzleRow(
                puzzle=str(puzzle),
                solution=str(solution),
                source="challenge100",
                name=f"challenge100_{puzzle_id}",
                puzzle_id=puzzle_id,
            )


def _iter_hf_parquet_rows(dataset_dir: Path, *, split: str, limit: Optional[int], batch_size: int) -> Iterator[PuzzleRow]:
    pattern = re.compile(rf"{re.escape(split)}_(\d+)\.parquet$")
    shard_paths = []
    for p in sorted(dataset_dir.iterdir(), key=lambda item: item.name):
        match = pattern.fullmatch(p.name)
        if match:
            shard_paths.append((int(match.group(1)), p))
    if not shard_paths:
        raise FileNotFoundError(f"Could not find parquet shards for split {split!r} under {dataset_dir}.")

    seen = 0
    for shard_idx, shard_path in shard_paths:
        parquet_file = pq.ParquetFile(shard_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["puzzle", "solution"]):
            batch_dict = batch.to_pydict()
            for puzzle, solution in zip(batch_dict["puzzle"], batch_dict["solution"]):
                if limit is not None and seen >= limit:
                    return
                yield PuzzleRow(
                    puzzle=puzzle,
                    solution=solution,
                    source="hf_sudoku_dataset",
                    name=f"sudoku_dataset_{split}_{seen}",
                    puzzle_id=f"{split}:{shard_idx}:{seen}",
                )
                seen += 1


def _build_task(row: PuzzleRow, *, num: int):
    initial_evidence = initial_board_into_evidence(row.puzzle, colNum=num, rowNum=num)
    solution_evidence = initial_board_into_evidence(row.solution, colNum=num, rowNum=num)
    task = build_constraint_sudoku_task(
        network_factory=lambda evidence: standard_sudoku.get_assignment_as_constraint_network(
            num=num,
            startAssignment=evidence,
        ),
        initial_evidence=initial_evidence,
        solution_evidence=solution_evidence,
        num=num,
        name=row.name,
    )
    task.metadata.update({"source": row.source, "puzzle_id": row.puzzle_id})
    return task


def _iter_rows(
    *,
    challenge100_csv: Optional[Path],
    challenge100_limit: Optional[int],
    hf_dataset_dir: Optional[Path],
    hf_split: str,
    hf_limit: Optional[int],
    hf_batch_size: int,
) -> Iterator[PuzzleRow]:
    if challenge100_csv is not None:
        yield from _iter_challenge100_rows(challenge100_csv, limit=challenge100_limit)
    if hf_dataset_dir is not None:
        yield from _iter_hf_parquet_rows(hf_dataset_dir, split=hf_split, limit=hf_limit, batch_size=hf_batch_size)


def prepare_filtered_parquet(
    *,
    out_parquet: Path,
    num: int,
    challenge100_csv: Optional[Path],
    challenge100_limit: Optional[int],
    hf_dataset_dir: Optional[Path],
    hf_split: str,
    hf_limit: Optional[int],
    hf_batch_size: int,
) -> None:
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    engine = AlphaReasonClosureEngine()
    kept = 0
    processed = 0

    # We keep a simple append-only journal so we can periodically rewrite a fully valid parquet
    # at `out_parquet` without producing multiple shard files.
    journal_path = out_parquet.with_suffix(out_parquet.suffix + ".jsonl")
    tmp_path = out_parquet.with_suffix(out_parquet.suffix + ".tmp")
    with journal_path.open("w"):
        pass

    def write_parquet_checkpoint() -> None:
        writer: Optional[pq.ParquetWriter] = None
        batch: list[dict] = []

        def flush_batch() -> None:
            nonlocal writer, batch
            if not batch:
                return
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema)
            writer.write_table(table)
            batch = []

        with journal_path.open("r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                batch.append(json.loads(line))
                if len(batch) >= 5000:
                    flush_batch()
            flush_batch()

        if writer is None:
            raise ValueError("No puzzles were kept; output parquet would be empty.")
        writer.close()
        os.replace(tmp_path, out_parquet)

    checkpoint_every_processed = 100
    journal_handle = journal_path.open("a", buffering=1)
    last_checkpoint_processed = 0
    for row in _iter_rows(
        challenge100_csv=challenge100_csv,
        challenge100_limit=challenge100_limit,
        hf_dataset_dir=hf_dataset_dir,
        hf_split=hf_split,
        hf_limit=hf_limit,
        hf_batch_size=hf_batch_size,
    ):
        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed}, kept {kept}...", flush=True)
        task = _build_task(row, num=num)
        state = engine.close(task, dict(task.initial_evidence), active_inference_clusters={})

        # Keep only puzzles that are NOT solved after one closure step and not contradictory.
        if (not state.solved) and (not state.contradiction):
            kept += 1
            record = {
                "puzzle": row.puzzle,
                "solution": row.solution,
                "source": row.source,
                "name": row.name,
                "puzzle_id": row.puzzle_id,
                "assigned_count": int(state.assigned_count),
                "message_count": int(getattr(state, "message_count", 0)),
            }
            journal_handle.write(json.dumps(record) + "\n")

        if processed - last_checkpoint_processed >= checkpoint_every_processed:
            journal_handle.flush()
            write_parquet_checkpoint()
            last_checkpoint_processed = processed

    journal_handle.flush()
    journal_handle.close()
    write_parquet_checkpoint()

    print(f"Processed {processed}, kept {kept}, wrote {out_parquet}")
    # On success, keep only the parquet output; the journal is only for mid-run checkpoints/resume.
    try:
        journal_path.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass


def main() -> int:
    # Edit these limits/paths directly; this script intentionally has no CLI flags.
    out_parquet = DEFAULT_OUT_PARQUET
    num = 3

    challenge100_csv = DEFAULT_CHALLENGE100_CSV
    challenge100_limit = 20

    hf_dataset_dir = DEFAULT_HF_DATASET_DIR
    hf_split = "train"
    hf_limit = 100000
    hf_batch_size = 256

    print("Preparing filtered training parquet (unsolved after one closure step):", flush=True)
    print("out_parquet:", out_parquet, flush=True)
    print("num:", num, flush=True)
    print("challenge100_csv:", challenge100_csv, flush=True)
    print("challenge100_limit:", challenge100_limit, flush=True)
    print("hf_dataset_dir:", hf_dataset_dir, flush=True)
    print("hf_split:", hf_split, flush=True)
    print("hf_limit:", hf_limit, flush=True)
    print("hf_batch_size:", hf_batch_size, flush=True)

    prepare_filtered_parquet(
        out_parquet=out_parquet,
        num=num,
        challenge100_csv=challenge100_csv,
        challenge100_limit=challenge100_limit,
        hf_dataset_dir=hf_dataset_dir,
        hf_split=hf_split,
        hf_limit=hf_limit,
        hf_batch_size=hf_batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
