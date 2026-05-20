from experiments.constraint_networks.sudoku_variant_forward_chaining import (
    ConstraintNetworkSudokuVariantForwardChainer,
    SudokuVariantHybridForwardChainingClosureEngine,
    SudokuVariantForwardChainer,
    SudokuVariantPropagationResult,
    all_cell_keys,
    cell_key,
)
from experiments.alphareason.task import AlphaReasonTask
from experiments.alphareason.tasks.sudoku import constraint_network_closure
from demonstrations.sudoku.examples import (
    introduction_to_killer_sudoku,
    sakana_fish,
    what_s_the_difference,
)
from tnreason import engine


def _cell_aux_parity_core(cell, aux):
    return engine.create_from_slice_iterator(
        colors=[cell, aux],
        shape=[9, 2],
        sliceIterator=[
            (1, {cell: value, aux: value % 2})
            for value in range(9)
        ],
    )


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


def test_hybrid_forward_chaining_feeds_binary_domains_into_generic_fallback():
    cell = cell_key(0, 1)
    task = AlphaReasonTask(
        name="hybrid_binary_generic",
        network_factory=lambda evidence: {
            "cell_aux": _cell_aux_parity_core(cell, "aux"),
        },
        closure_function=constraint_network_closure,
        initial_evidence={},
        target_feature_keys=("aux",),
        action_feature_keys=(cell, "aux"),
        feature_domain_sizes={cell: 9, "aux": 2},
        assignment_evidence_map={
            cell: {value: {cell: value} for value in range(9)},
            "aux": {value: {"aux": value} for value in range(2)},
        },
        consistency_checker=lambda constraint_network, assignment: True,
    )

    class FixedBinaryDomainEngine(SudokuVariantHybridForwardChainingClosureEngine):
        def _binary_chainer(self, core_dict):
            class FixedBinaryChainer:
                fully_supported = False

                def propagate(self, evidence):
                    return SudokuVariantPropagationResult(
                        domains={cell: {1, 3, 5, 7}},
                        contradiction=False,
                        iterations=1,
                    )

            return FixedBinaryChainer()

    state = FixedBinaryDomainEngine().close(task, task.initial_evidence)

    assert not state.contradiction
    assert state.feature_supports[cell] == (1, 3, 5, 7)
    assert state.target_assignments == {"aux": 1}


def test_hybrid_forward_chaining_skips_generic_fallback_for_supported_binary_network():
    core_dict = sakana_fish.get_assignment_as_constraint_network(num=3, startAssignment={})
    chainer = ConstraintNetworkSudokuVariantForwardChainer(core_dict, num=3)

    assert chainer.fully_supported


def test_constraint_network_chainer_uses_num_when_compiling_n2_binary_cores():
    core_dict = introduction_to_killer_sudoku.get_assignment_as_constraint_network(
        num=2,
        startAssignment={},
        include_initial_board=False,
    )
    chainer = ConstraintNetworkSudokuVariantForwardChainer(core_dict, num=2)
    valid_cell_keys = set(all_cell_keys(num=2))

    assert chainer.binary_constraints
    assert all(
        left in valid_cell_keys and right in valid_cell_keys
        for left, right, _ in chainer.binary_constraints
    )


def test_constraint_network_chainer_compiles_nary_variant_cores():
    core_dict = what_s_the_difference.get_assignment_as_constraint_network(
        num=3,
        startAssignment={},
        include_initial_board=False,
    )
    chainer = ConstraintNetworkSudokuVariantForwardChainer(core_dict, num=3)

    assert chainer.fully_supported
    assert len(chainer.table_constraints) == 13
