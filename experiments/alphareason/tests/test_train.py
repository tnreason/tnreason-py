from pathlib import Path

from ..train import (
    TrainingConfig,
    build_demo_training_tasks,
    load_hf_sudoku_dataset_training_tasks,
    load_challenge100_training_tasks,
    train_supervised,
)


def test_train_supervised_runs_end_to_end(tmp_path):
    checkpoint_path = tmp_path / "alphareason_demo.pt"
    tasks = build_demo_training_tasks()

    model, history = train_supervised(
        tasks,
        config=TrainingConfig(
            epochs=2,
            learning_rate=1e-3,
            max_samples_per_task=3,
            checkpoint_path=str(checkpoint_path),
            seed=0,
        ),
    )

    assert model is not None
    assert len(history) == 2
    assert history[-1]["samples"] > 0
    assert checkpoint_path.exists()


def test_load_challenge100_training_tasks_builds_real_tasks():
    tasks = load_challenge100_training_tasks(limit=2, offset=0)

    assert len(tasks) == 2
    assert all(task.name.startswith("challenge100_") for task in tasks)
    assert all(task.metadata["source"] == "challenge100" for task in tasks)
    assert all(task.solution_assignments for task in tasks)


def test_load_hf_sudoku_dataset_training_tasks_builds_real_tasks():
    tasks = load_hf_sudoku_dataset_training_tasks(limit=2, offset=0, batch_size=2)

    assert len(tasks) == 2
    assert all(task.name.startswith("sudoku_dataset_train_") for task in tasks)
    assert all(task.metadata["source"] == "hf_sudoku_dataset" for task in tasks)
    assert all(task.solution_assignments for task in tasks)
