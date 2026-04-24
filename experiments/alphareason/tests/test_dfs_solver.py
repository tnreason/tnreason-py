from ..closure_engine import AlphaReasonClosureEngine
from ..dfs_solver import AlphaReasonDFSSolver
from ..run_experiment import build_demo_constraint_sudoku_task


def test_dfs_solver_solves_demo_constraint_sudoku():
    task = build_demo_constraint_sudoku_task()

    result = AlphaReasonDFSSolver(
        closure_engine=AlphaReasonClosureEngine(),
        max_nodes=1000,
    ).solve(task)

    assert result.solved
    assert not result.state.contradiction
    assert result.state.assigned_count == len(task.target_feature_keys)
    assert result.nodes <= 1000
