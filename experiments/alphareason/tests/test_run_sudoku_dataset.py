from experiments.alphareason.run_sudoku_dataset import DEFAULT_DATASET, run_dataset


def test_run_sudoku_dataset_limit_one_closure_only():
    rows = run_dataset(dataset_path=DEFAULT_DATASET, limit=1, max_steps=0, simulations=0)

    assert len(rows) == 1
    assert rows[0]["target_count"] == 81
    assert rows[0]["history_length"] == 0
    assert rows[0]["cache_misses"] >= 1

