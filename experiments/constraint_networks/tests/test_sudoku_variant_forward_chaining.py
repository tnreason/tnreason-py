from experiments.constraint_networks.sudoku_variant_forward_chaining import (
    ConstraintNetworkSudokuVariantForwardChainer,
    SudokuVariantForwardChainer,
    cell_key,
)
from demonstrations.sudoku.examples import sakana_fish


def test_sudoku_variant_forward_chainer_accepts_custom_parity_edges():
    chainer = SudokuVariantForwardChainer(num=3, parity_edges=[((0, 0), (0, 1))])

    result = chainer.propagate(
        {
            cell_key(0, 0): 0,
        }
    )

    assert not result.contradiction
    assert result.domains[cell_key(0, 0)] == {0}
    assert result.domains[cell_key(0, 1)] == {1, 3, 5, 7}


def test_sudoku_variant_forward_chainer_detects_custom_parity_conflict():
    chainer = SudokuVariantForwardChainer(num=3, parity_edges=[((0, 0), (0, 1))])

    result = chainer.propagate(
        {
            cell_key(0, 0): 0,
            cell_key(0, 1): 2,
        }
    )

    assert result.contradiction


def test_constraint_network_sudoku_variant_chainer_compiles_sakana_edges():
    core_dict = sakana_fish.get_assignment_as_constraint_network(num=3, startAssignment={})
    chainer = ConstraintNetworkSudokuVariantForwardChainer(core_dict, num=3)

    result = chainer.propagate(
        {
            cell_key(0, 0): 0,
        }
    )

    assert not result.contradiction
    assert result.domains[cell_key(1, 0)] == {1}


def test_constraint_network_sudoku_variant_chainer_compiles_odd_indicator_edges():
    core_dict = sakana_fish.get_assignment_as_constraint_network(num=3, startAssignment={})
    chainer = ConstraintNetworkSudokuVariantForwardChainer(core_dict, num=3)

    result = chainer.propagate(
        {
            cell_key(2, 1): 0,
        }
    )

    assert not result.contradiction
    assert result.domains[cell_key(2, 2)] == {1, 3, 5, 7}
