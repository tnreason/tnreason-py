from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence
from demonstrations.sudoku.examples import standard_sudoku

from ..run_experiment import (
    build_demo_constraint_sudoku_task,
    build_n3_example_constraint_sudoku_task,
    run_sakana_fish_constraint_sudoku,
    solve_task,
)
from ..tasks.sudoku import (
    build_constraint_sudoku_task,
    build_sakana_fish_constraint_sudoku_task,
    constraint_network_closure,
)


def test_build_constraint_sudoku_task_extracts_targets_and_solution():
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

    task = build_constraint_sudoku_task(
        network_factory=lambda evidence: standard_sudoku.get_assignment_as_constraint_network(
            num=3,
            startAssignment=evidence,
        ),
        initial_evidence=initial_evidence,
        solution_evidence=solution_evidence,
        num=3,
    )

    assert task.closure_function is constraint_network_closure
    assert len(task.target_feature_keys) == 81
    assert task.feature_domain_sizes["pos_0_0_0_0"] == 9
    assert task.solution_assignments["pos_0_0_0_0"] == 0
    assert task.assignment_to_evidence("pos_0_0_0_0", 0) == {"a_0_0_0_0_0": 1}


def test_constraint_sudoku_demo_runs_through_solver():
    task = build_demo_constraint_sudoku_task()

    state, history, cache_info = solve_task(task, simulations=8, max_steps=8)

    assert not state.contradiction
    assert state.assigned_count >= 4
    assert isinstance(history, list)
    assert cache_info["misses"] >= 1


def test_constraint_sudoku_n3_example_solves():
    task = build_n3_example_constraint_sudoku_task()

    state, history, cache_info = solve_task(task, simulations=5, max_steps=5)

    assert state.solved
    assert not state.contradiction
    assert state.assigned_count == len(task.target_feature_keys)
    assert history == []
    assert cache_info["misses"] >= 1


def test_sakana_fish_constraint_sudoku_task_runs_through_solver():
    task = build_sakana_fish_constraint_sudoku_task()

    state, history, cache_info = solve_task(task, simulations=2, max_steps=2)

    assert not state.contradiction
    assert isinstance(history, list)
    assert cache_info["misses"] >= 1


def test_run_sakana_fish_constraint_sudoku_helper():
    state, history, cache_info = run_sakana_fish_constraint_sudoku(simulations=1, max_steps=1)

    assert not state.contradiction
    assert isinstance(history, list)
    assert cache_info["misses"] >= 1
