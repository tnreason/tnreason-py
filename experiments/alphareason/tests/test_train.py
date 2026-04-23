from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..data_loaders import (
    REPO_CHALLENGE100_CSV_PATH,
    REPO_HF_SUDOKU_DATASET_DIR,
    load_challenge100_training_tasks,
    load_hf_sudoku_dataset_training_tasks,
    load_prepared_sudoku_training_tasks,
)
from ..run_experiment import build_demo_sudoku_task
from ..train import (
    train_supervised,
)


def test_train_supervised_runs_end_to_end(tmp_path):
    checkpoint_path = tmp_path / "alphareason_demo.pt"
    tasks = [
        build_demo_sudoku_task(),
        build_demo_sudoku_task(
            initial="1.3." ".4.." "..4." "4..1",
            name="demo_standard_sudoku_4x4_variant_a",
        ),
        build_demo_sudoku_task(
            initial="..3." "34.." "..4." "4...",
            name="demo_standard_sudoku_4x4_variant_b",
        ),
    ]

    model, history = train_supervised(
        tasks,
        epochs=2,
        learning_rate=1e-3,
        checkpoint_path=str(checkpoint_path),
        seed=0,
    )

    assert model is not None
    assert len(history) == 2
    assert history[-1]["samples"] > 0
    assert checkpoint_path.exists()


def test_load_hf_sudoku_dataset_training_tasks_builds_real_tasks():
    tasks = load_hf_sudoku_dataset_training_tasks(
        num_samples=2,
        batch_size=2,
        dataset_dir=str(REPO_HF_SUDOKU_DATASET_DIR),
    )

    assert len(tasks) == 2
    assert all(task.name.startswith("sudoku_dataset_train_") for task in tasks)
    assert all(task.metadata["source"] == "hf_sudoku_dataset" for task in tasks)
    assert all(task.solution_assignments for task in tasks)


def test_load_challenge100_training_tasks_builds_real_tasks():
    tasks = load_challenge100_training_tasks(num_samples=2, csv_path=str(REPO_CHALLENGE100_CSV_PATH), num=3)

    assert len(tasks) == 2
    assert all(task.name.startswith("challenge100_") for task in tasks)
    assert all(task.metadata["source"] == "challenge100" for task in tasks)
    assert all(task.solution_assignments for task in tasks)


def test_load_prepared_sudoku_training_tasks_reads_parquet(tmp_path):
    parquet_path = tmp_path / "prepared.parquet"
    table = pa.Table.from_pylist(
        [
            {
                "puzzle": "..3..5....7.3...2.8...6.4...1...8..5..8.2.6..3..9...1...6.7...9.2...9.4....4..1..",
                "solution": "243815967675394821891762453912648735758123694364957218436271589127589346589436172",
                "source": "challenge100",
                "puzzle_id": "nhk2311h1",
                "name": "challenge100_nhk2311h1",
            }
        ]
    )
    pq.write_table(table, parquet_path)

    tasks = load_prepared_sudoku_training_tasks(parquet_path=str(parquet_path), num_samples=1, num=3)
    assert len(tasks) == 1
    assert tasks[0].name == "challenge100_nhk2311h1"
    assert tasks[0].metadata["prepared"] is True
    assert tasks[0].solution_assignments
