from experiments.alphareason.dfs_solver import AlphaReasonDFSSolver
from experiments.alphareason.tasks.sudoku import build_sakana_fish_constraint_sudoku_task
from experiments.constraint_networks.sakana_fish_forward_chaining import (
    SakanaFishForwardChainer,
    SakanaFishForwardChainingClosureEngine,
)


def test_sakana_fish_forward_chainer_root_domains_are_supported():
    chainer = SakanaFishForwardChainer()

    result = chainer.propagate({})

    assert not result.contradiction
    assert all(result.domains[feature_key] for feature_key in chainer.cell_keys)
    assert any(len(result.domains[feature_key]) < 9 for feature_key in chainer.cell_keys)


def test_sakana_fish_forward_chainer_detects_bad_white_dot_assignment():
    chainer = SakanaFishForwardChainer()

    result = chainer.propagate(
        {
            "pos_0_0_0_0": 0,
            "pos_0_1_0_0": 2,
        }
    )

    assert result.contradiction


def test_sakana_fish_forward_chainer_solves_with_dfs():
    task = build_sakana_fish_constraint_sudoku_task()
    solver = AlphaReasonDFSSolver(
        closure_engine=SakanaFishForwardChainingClosureEngine(),
        max_nodes=100,
        branch_feature_count=1,
        trace=False,
    )

    result = solver.solve(task)

    assert result.solved
    assert not result.state.contradiction
    assert result.state.assigned_count == len(task.target_feature_keys)
    assert result.state.target_assignments == task.solution_assignments
