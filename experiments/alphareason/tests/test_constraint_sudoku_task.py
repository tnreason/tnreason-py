from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence
from demonstrations.sudoku.examples import standard_sudoku
from tnreason import engine

from ..run_experiment import (
    build_demo_constraint_sudoku_task,
    build_n3_example_constraint_sudoku_task,
    run_sakana_fish_constraint_sudoku,
    solve_task,
)
from ..closure_engine import AlphaReasonClosureEngine
from ..task import AlphaReasonTask
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
    assert len(task.action_feature_keys) == len(task.target_feature_keys)
    assert all(not feature_key.startswith("a_") for feature_key in task.action_feature_keys)
    assert all(not feature_key.startswith(("row_", "col_", "square_")) for feature_key in task.action_feature_keys)
    assert task.feature_domain_sizes["pos_0_0_0_0"] == 9
    assert task.solution_assignments["pos_0_0_0_0"] == 0
    assert task.assignment_to_evidence("pos_0_0_0_0", 0) == {
        "pos_0_0_0_0": 0,
        "a_0_0_0_0_0": 1,
    }


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


def test_sakana_fish_root_actions_are_cell_assignments():
    task = build_sakana_fish_constraint_sudoku_task()

    state = AlphaReasonClosureEngine().close(task, task.initial_evidence)

    assert state.legal_actions
    assert all(action.feature_key.startswith("pos_") for action in state.legal_actions)


def test_run_sakana_fish_constraint_sudoku_helper():
    state, history, cache_info = run_sakana_fish_constraint_sudoku(simulations=1, max_steps=1)

    assert not state.contradiction
    assert isinstance(history, list)
    assert cache_info["misses"] >= 1


def test_constraint_closure_preserves_assigned_target_missing_from_network():
    task = AlphaReasonTask(
        name="missing_target",
        network_factory=lambda evidence: {},
        closure_function=constraint_network_closure,
        initial_evidence={"x_is_1": 1},
        target_feature_keys=("x",),
        feature_domain_sizes={"x": 2},
        assignment_evidence_map={
            "x": {
                0: {"x_is_0": 1},
                1: {"x_is_1": 1},
            }
        },
        consistency_checker=lambda constraint_network, assignment: True,
    )

    state = AlphaReasonClosureEngine().close(task, task.initial_evidence)

    assert state.target_assignments == {"x": 1}
    assert state.assigned_count == 1
    assert state.solved


def test_constraint_closure_keeps_local_tensor_priors_after_gac_masking():
    task = AlphaReasonTask(
        name="weighted_unary",
        network_factory=lambda evidence: {
            "x_weight": engine.create_from_slice_iterator(
                colors=["x"],
                shape=[3],
                sliceIterator=[
                    (3, {"x": 0}),
                    (1, {"x": 1}),
                ],
            ),
        },
        closure_function=constraint_network_closure,
        initial_evidence={},
        target_feature_keys=("x",),
        feature_domain_sizes={"x": 3},
        assignment_evidence_map={
            "x": {
                0: {"x": 0},
                1: {"x": 1},
                2: {"x": 2},
            },
        },
        consistency_checker=lambda constraint_network, assignment: True,
    )

    state = AlphaReasonClosureEngine().close(task, task.initial_evidence)

    assert state.feature_supports["x"] == (0, 1)
    assert state.feature_priors["x"] == (0.75, 0.25, 0.0)
