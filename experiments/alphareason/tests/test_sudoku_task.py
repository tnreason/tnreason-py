from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence

from ..tasks.sudoku import build_standard_sudoku_task


def test_build_standard_sudoku_task_extracts_targets_and_solution():
    initial_evidence = initial_board_into_evidence("1" + "." * 80)
    solution_evidence = initial_board_into_evidence(
        "123456789"
        "456789123"
        "789123456"
        "214365897"
        "365897214"
        "897214365"
        "531642978"
        "642978531"
        "978531642"
    )

    task = build_standard_sudoku_task(initial_evidence=initial_evidence, solution_evidence=solution_evidence, num=3)

    assert len(task.target_feature_keys) == 81
    assert task.feature_domain_sizes["pos_0_0_0_0"] == 9
    assert task.solution_assignments["pos_0_0_0_0"] == 0
    assert task.assignment_to_evidence("pos_0_0_0_0", 0) == {"a_0_0_0_0_0": 1}

