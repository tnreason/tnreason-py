from ..dataset import build_supervised_dataset
from ..run_experiment import build_demo_sudoku_task


def test_build_supervised_dataset_extracts_samples():
    task = build_demo_sudoku_task()
    dataset = build_supervised_dataset([task], max_samples_per_task=4, shuffle_good_actions=False, seed=0)

    assert dataset
    assert all(sample.task_name == task.name for sample in dataset)
    assert all(sample.policy_target for sample in dataset)
    assert all(sample.value_target == 1.0 for sample in dataset)

