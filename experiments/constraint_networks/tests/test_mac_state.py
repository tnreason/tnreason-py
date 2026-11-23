from itertools import product
import numpy as np
import pytest

from tnreason import engine, representation

from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.generalized_arc_consistency import (
    enforce_generalized_arc_consistency,
)


def _relation_core(name, colors, shape, tuples, weights=None):
    if weights is None:
        weights = [1] * len(tuples)
    return engine.create_from_slice_iterator(
        name=name,
        colors=list(colors),
        shape=list(shape),
        sliceIterator=[
            (weight, dict(zip(colors, values)))
            for weight, values in zip(weights, tuples)
        ],
    )


def _equal_core(name, left, right, dimension=2):
    return _relation_core(
        name,
        (left, right),
        (dimension, dimension),
        tuple((value, value) for value in range(dimension)),
    )


def _not_equal_core(name, left, right, dimension):
    return _relation_core(
        name,
        (left, right),
        (dimension, dimension),
        tuple(
            (left_value, right_value)
            for left_value in range(dimension)
            for right_value in range(dimension)
            if left_value != right_value
        ),
    )


def _assert_domain_tensors(state, tensors=None):
    for color, domain in state.domains.items():
        tensor = state._domain_tensors[color]
        assert tensor.dtype == np.bool_ and tensor.shape == (state.dimensions[color],)
        assert tuple(np.flatnonzero(tensor)) == domain
        if tensors is not None:
            assert tensor is tensors[color]


def test_mac_initial_propagation_matches_gac_equality_chain():
    core_dict = {
        "xy": _equal_core("xy", "x", "y"),
        "yz": _equal_core("yz", "y", "z"),
        "z_is_1": representation.create_basis_core(
            name="z_is_1",
            shape=[2],
            colors=["z"],
            numberTuple=[1],
        ),
    }
    expected_domains, expected_contradiction = enforce_generalized_arc_consistency(core_dict)

    network = ConstraintNetwork(core_dict)
    state = network.create_mac_state()
    result = state.propagate()

    assert result
    assert not expected_contradiction
    assert state.snapshot_domains() == expected_domains
    assert state.snapshot_domains() == {"x": (1,), "y": (1,), "z": (1,)}


def test_mac_nested_levels_restore_domains_queue_and_contradiction():
    network = ConstraintNetwork({"xy": _equal_core("xy", "x", "y")})
    state = network.create_mac_state()
    tensors = dict(state._domain_tensors)
    _assert_domain_tensors(state, tensors)

    root_result = state.propagate()
    root_domains = state.snapshot_domains()
    assert root_result
    assert root_domains == {"x": (0, 1), "y": (0, 1)}
    _assert_domain_tensors(state, tensors)

    state.push_level()
    state.assign("x", 1)
    assert state.propagate()
    assert state.snapshot_domains() == {"x": (1,), "y": (1,)}
    _assert_domain_tensors(state, tensors)

    state.push_level()
    state.assign("y", 0)
    assert state.contradiction
    assert not state.propagate()
    _assert_domain_tensors(state, tensors)

    state.pop_level()
    assert not state.contradiction
    assert state.snapshot_domains() == {"x": (1,), "y": (1,)}
    assert state.propagate().queue_pops == 0
    _assert_domain_tensors(state, tensors)

    state.pop_level()
    assert not state.contradiction
    assert state.snapshot_domains() == root_domains
    _assert_domain_tensors(state, tensors)

    # A sibling branch must not inherit domains or queue entries from x=1.
    state.push_level()
    state.assign("x", 0)
    assert state.propagate()
    assert state.snapshot_domains() == {"x": (0,), "y": (0,)}
    _assert_domain_tensors(state, tensors)
    state.pop_level()
    assert state.snapshot_domains() == root_domains
    _assert_domain_tensors(state, tensors)


def test_mac_detects_contradiction_and_is_idempotent_at_fixed_point():
    core_dict = {
        "xy": _equal_core("xy", "x", "y"),
        "x_is_0": representation.create_basis_core(
            name="x_is_0", shape=[2], colors=["x"], numberTuple=[0]
        ),
        "y_is_1": representation.create_basis_core(
            name="y_is_1", shape=[2], colors=["y"], numberTuple=[1]
        ),
    }
    state = ConstraintNetwork(core_dict).create_mac_state()

    first = state.propagate()
    second = state.propagate()

    assert first.contradiction
    assert not first
    assert second.contradiction
    assert second.queue_pops == 0
    assert any(not domain for domain in state.domains.values())


def test_mac_handles_genuine_ternary_constraint_and_rollback():
    ternary = _relation_core(
        "xyz",
        ("x", "y", "z"),
        (2, 2, 2),
        ((0, 0, 0), (1, 1, 0), (1, 1, 1)),
    )
    state = ConstraintNetwork({"xyz": ternary}).create_mac_state()
    assert state.propagate()

    state.push_level()
    state.assign("z", 1)
    assert state.propagate()
    assert state.snapshot_domains() == {"x": (1,), "y": (1,), "z": (1,)}
    state.pop_level()

    assert state.snapshot_domains() == {
        "x": (0, 1),
        "y": (0, 1),
        "z": (0, 1),
    }


def test_mac_contracts_noncontiguous_domains_without_indexed_copies(monkeypatch):
    core = _relation_core("xyz", ("x", "y", "z"), (5, 6, 4),
                          ((1, 0, 2), (3, 4, 2)), weights=(1, -1))
    domains = {"x": (1, 3, 4), "y": (0, 2, 4), "z": (1, 2)}
    expected, contradiction = enforce_generalized_arc_consistency({"xyz": core}, dict(domains))
    state = ConstraintNetwork({"xyz": core}).create_mac_state(domains)
    tensors = dict(state._domain_tensors)
    _assert_domain_tensors(state, tensors)

    def unexpected_indexing(*args):
        raise AssertionError("NumPy propagation must contract persistent domain tensors directly.")

    monkeypatch.setattr(np, "ix_", unexpected_indexing)
    result = state.propagate()
    assert not contradiction and result
    assert state.snapshot_domains() == expected == {"x": (1, 3), "y": (0, 4), "z": (2,)}
    assert result.domain_reductions == 3 and result.revisions == 3
    assert result.queue_pops == 1
    _assert_domain_tensors(state, tensors)

    for value, partner in ((3, 4), (1, 0)):
        state.push_level()
        state.assign("x", value)
        assert state.propagate()
        assert state.snapshot_domains() == {"x": (value,), "y": (partner,), "z": (2,)}
        _assert_domain_tensors(state, tensors)
        state.pop_level()
        assert state.snapshot_domains() == expected
        _assert_domain_tensors(state, tensors)


@pytest.mark.parametrize("domain, expected", [((0, 2, 4), (2, 4)), ((4,), (4,)), ((0,), ())])
def test_mac_batch_handles_unary_domains(domain, expected):
    core = _relation_core("unary", ("x",), (5,), ((2,), (4,)), weights=(1, -1))
    state = ConstraintNetwork({"unary": core}).create_mac_state({"x": domain})
    result = state.propagate()
    assert result.contradiction == (not expected)
    assert state.domains["x"] == expected
    _assert_domain_tensors(state)


def test_mac_domain_tensors_handle_initial_and_branch_empty_domains():
    network = ConstraintNetwork({"xy": _equal_core("xy", "x", "y")})
    empty = network.create_mac_state({"x": ()})
    tensors = dict(empty._domain_tensors)
    assert not empty.propagate()
    _assert_domain_tensors(empty, tensors)

    state = network.create_mac_state({"x": (1,)})
    tensors = dict(state._domain_tensors)
    assert state.propagate()
    root = state.snapshot_domains()
    state.push_level()
    assert state.restrict_domain("y", ())
    assert not state.propagate()
    _assert_domain_tensors(state, tensors)
    state.pop_level()
    assert state.snapshot_domains() == root and state.propagate()
    _assert_domain_tensors(state, tensors)


def test_mac_domain_tensors_restore_after_propagation_fails_in_sibling_branches():
    cores = {left + right: _not_equal_core(left + right, left, right, 2)
             for left, right in (("a", "b"), ("b", "c"), ("c", "a"))}
    state = ConstraintNetwork(cores).create_mac_state()
    tensors = dict(state._domain_tensors)
    assert state.propagate()
    root = state.snapshot_domains()
    for value in (0, 1):
        state.push_level()
        state.assign("a", value)
        assert not state.propagate()
        _assert_domain_tensors(state, tensors)
        state.pop_level()
        assert not state.contradiction and state.snapshot_domains() == root
        _assert_domain_tensors(state, tensors)


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8, np.int64, np.float64])
def test_mac_contractions_accept_boolean_integer_and_float_support_tensors(dtype):
    core = _relation_core("xy", ("x", "y"), (3, 4), ((0, 1), (1, 2), (2, 3)))
    core.values = core.values.astype(dtype)
    state = ConstraintNetwork({"xy": core}).create_mac_state({"x": (0, 2), "y": (1, 2)})
    assert state.propagate()
    assert state.snapshot_domains() == {"x": (0,), "y": (1,)}
    _assert_domain_tensors(state)


def test_mac_binary_contractions_do_not_overflow_many_supports():
    core = _relation_core("xy", ("x", "y"), (2, 256),
                          tuple((0, value) for value in range(256)))
    core.values = core.values.astype(np.uint8)
    state = ConstraintNetwork({"xy": core}).create_mac_state()
    assert state.propagate()
    assert state.domains["x"] == (0,) and state.domains["y"] == tuple(range(256))
    _assert_domain_tensors(state)


def test_mac_ternary_contractions_use_realized_nonzero_support_not_signed_sums():
    core = _relation_core("xyz", ("x", "y", "z"), (2, 2, 2),
                          ((0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 0)),
                          weights=(1, -1, 1, -1))
    state = ConstraintNetwork({"xyz": core}).create_mac_state()
    assert state.propagate()
    assert state.snapshot_domains() == {"x": (0,), "y": (0,), "z": (0, 1)}
    _assert_domain_tensors(state)


@pytest.mark.parametrize("arity", [4, 31])
@pytest.mark.parametrize("weight", [0, -1])
def test_mac_preserves_high_arity_support_at_einsum_operand_limit(arity, weight):
    colors = tuple(f"x{idx}" for idx in range(arity))
    core = _relation_core("high_arity", colors, (1,) * arity, ((0,) * arity,), weights=(weight,))
    state = ConstraintNetwork({"high_arity": core}).create_mac_state()
    assert state.propagate().contradiction == (weight == 0)
    if weight:
        assert state.snapshot_domains() == {color: (0,) for color in colors}
    _assert_domain_tensors(state)


def test_mac_treats_nonzero_weights_as_constraint_support():
    # Arithmetic contraction would cancel the two supports for x=0.  CSP
    # propagation must use nonzero support rather than the original weights.
    weighted = _relation_core(
        "weighted",
        ("x", "y"),
        (2, 2),
        ((0, 0), (0, 1), (1, 0)),
        weights=(1, -1, 1),
    )
    state = ConstraintNetwork({"weighted": weighted}).create_mac_state()

    assert state.propagate()
    assert state.snapshot_domains() == {"x": (0, 1), "y": (0, 1)}


def test_mac_states_are_isolated_and_do_not_modify_constraint_cores():
    equality = _equal_core("xy", "x", "y")
    original_values = np.asarray(equality.values).copy()
    network = ConstraintNetwork({"xy": equality})
    left = network.create_mac_state()
    right = network.create_mac_state()
    left_tensors, right_tensors = dict(left._domain_tensors), dict(right._domain_tensors)
    assert all(not np.shares_memory(left_tensors[color], right_tensors[color]) for color in left_tensors)
    left.propagate()
    right.propagate()

    exported_domain_core = right.domain_cores["x"]
    exported_domain_core.values[:] = 0

    left.push_level()
    left.assign("x", 1)
    left.propagate()

    assert left.domains["y"] == (1,)
    assert right.domains["x"] == (0, 1)
    assert right.domains["y"] == (0, 1)
    assert right.domain_cores["x"].values.sum() == 2
    assert np.array_equal(equality.values, original_values)
    _assert_domain_tensors(left, left_tensors)
    _assert_domain_tensors(right, right_tensors)


def test_mac_state_snapshots_support_but_new_state_sees_core_value_mutation():
    relation = _equal_core("xy", "x", "y")
    network = ConstraintNetwork({"xy": relation})
    equality_state = network.create_mac_state(domains={"x": (0,)})

    relation.values[:] = np.asarray([[0, 1], [1, 0]])
    inequality_state = network.create_mac_state(domains={"x": (0,)})

    assert equality_state.propagate()
    assert inequality_state.propagate()
    assert equality_state.domains["y"] == (0,)
    assert inequality_state.domains["y"] == (1,)


def test_mac_pop_restores_a_nonempty_root_queue():
    network = ConstraintNetwork({"xy": _equal_core("xy", "x", "y")})
    state = network.create_mac_state()

    state.push_level()
    state.assign("x", 1)
    assert state.propagate()
    state.pop_level()

    result = state.propagate()
    assert result.queue_pops > 0
    assert state.snapshot_domains() == {"x": (0, 1), "y": (0, 1)}


def test_mac_search_avoids_tensor_work_and_index_rebuilds_between_branches(monkeypatch):
    network = ConstraintNetwork({"xy": _equal_core("xy", "x", "y")}, maintain_indices=True)

    def unexpected_tensor_work(*args, **kwargs):
        raise AssertionError("MAC search must reuse its constraints, indices, and domain tuples.")

    monkeypatch.setattr(engine, "contract", unexpected_tensor_work)
    monkeypatch.setattr(engine, "create_from_slice_iterator", unexpected_tensor_work)
    monkeypatch.setattr(network, "rebuild_indices", unexpected_tensor_work)

    state = network.create_mac_state()
    assert state.propagate()
    for value in (0, 1):
        state.push_level()
        state.assign("x", value)
        assert state.propagate()
        assert state.domains["y"] == (value,)
        state.pop_level()

    assert network.solve_mac().solved


@pytest.mark.parametrize("core_type", ["NumpyCore", "PandasCore", "PolynomialCore"])
def test_mac_accepts_constraint_core_backends_and_initial_domains(core_type):
    equality = engine.create_from_slice_iterator(
        name="xy",
        colors=["x", "y"],
        shape=[2, 2],
        sliceIterator=[(1, {"x": 0, "y": 0}), (1, {"x": 1, "y": 1})],
        coreType=core_type,
    )

    state = ConstraintNetwork({"xy": equality}).create_mac_state(
        domains={"x": (1,)}
    )

    assert state.propagate()
    assert state.snapshot_domains() == {"x": (1,), "y": (1,)}


@pytest.mark.parametrize("core_type", ["PandasCore", "PolynomialCore"])
def test_sparse_gac_and_mac_check_realized_support_after_slice_cancellation(core_type):
    core = engine.create_from_slice_iterator(
        name="cancelled", colors=["x", "y"], shape=[2, 2], coreType=core_type,
        sliceIterator=[(1, {}), (-1, {"x": 0}), (-1, {"x": 1, "y": 0})])
    cores = {"cancelled": core}

    expected_domains, contradiction = enforce_generalized_arc_consistency(cores)
    state = ConstraintNetwork(cores).create_mac_state()

    assert not contradiction and state.propagate()
    assert state.snapshot_domains() == expected_domains == {"x": (1,), "y": (1,)}


def test_mac_validates_state_operations():
    state = ConstraintNetwork({"xy": _equal_core("xy", "x", "y")}).create_mac_state()

    with pytest.raises(KeyError):
        state.assign("missing", 0)
    with pytest.raises(ValueError):
        state.assign("x", 2)
    with pytest.raises(TypeError):
        state.restrict_domain("x", (0.5,))
    with pytest.raises(TypeError):
        state.restrict_domain("x", (True,))
    with pytest.raises(RuntimeError):
        state.pop_level()
    with pytest.raises(KeyError):
        ConstraintNetwork({"xy": _equal_core("xy", "x", "y")}).create_mac_state(
            {"missing": (0,)}
        )


def test_generic_mac_solver_works_for_graph_coloring():
    triangle = ConstraintNetwork(
        {
            "ab": _not_equal_core("ab", "a", "b", dimension=3),
            "bc": _not_equal_core("bc", "b", "c", dimension=3),
            "ca": _not_equal_core("ca", "c", "a", dimension=3),
        }
    )

    result = triangle.solve_mac()

    assert result.solved
    assert result.assignment is not None
    assert result.assignment["a"] != result.assignment["b"]
    assert result.assignment["b"] != result.assignment["c"]
    assert result.assignment["c"] != result.assignment["a"]


def test_generic_mac_solver_reports_unsatisfiable_network():
    two_color_triangle = ConstraintNetwork(
        {
            "ab": _not_equal_core("ab", "a", "b", dimension=2),
            "bc": _not_equal_core("bc", "b", "c", dimension=2),
            "ca": _not_equal_core("ca", "c", "a", dimension=2),
        }
    )

    result = two_color_triangle.solve_mac()

    assert not result.solved
    assert result.unsatisfiable
    assert result.assignment is None
    assert not result.node_limit_reached
    assert not result.root_contradiction


def test_generic_mac_solver_distinguishes_node_limit_from_unsatisfiable():
    network = ConstraintNetwork(
        {"xy": _not_equal_core("xy", "x", "y", dimension=2)}
    )

    result = network.solve_mac(max_nodes=1)

    assert not result.solved
    assert not result.unsatisfiable
    assert result.node_limit_reached
    assert not result.root_contradiction


def test_mac_handles_nullary_constraints():
    class ScalarCoreWithoutValues:
        colors = ()
        shape = ()

        def __getitem__(self, _item):
            return 0

    satisfiable = engine.create_from_slice_iterator(
        name="true",
        colors=[],
        shape=[],
        sliceIterator=[(1, {})],
    )
    unsatisfiable = ScalarCoreWithoutValues()

    true_result = ConstraintNetwork({"true": satisfiable}).solve_mac()
    false_result = ConstraintNetwork({"false": unsatisfiable}).solve_mac()

    assert true_result.solved
    assert not true_result.unsatisfiable
    assert true_result.assignment == {}
    assert not false_result.solved
    assert false_result.unsatisfiable
    assert false_result.root_contradiction


def test_random_small_networks_match_reference_gac_at_root_and_branch():
    rng = np.random.default_rng(20261005)

    for case_index in range(50):
        variable_count = int(rng.integers(2, 6))
        variables = tuple(f"x{index}" for index in range(variable_count))
        dimensions = {
            variable: int(rng.integers(2, 4))
            for variable in variables
        }
        core_dict = {}
        constraint_count = int(rng.integers(1, 7))
        for constraint_index in range(constraint_count):
            arity = int(rng.integers(1, min(5, variable_count) + 1))
            colors = tuple(rng.choice(variables, size=arity, replace=False).tolist())
            shape = tuple(dimensions[color] for color in colors)
            allowed = tuple(
                index
                for index in np.ndindex(shape)
                if rng.random() < 0.4
            )
            core_key = f"case_{case_index}_constraint_{constraint_index}"
            core_dict[core_key] = _relation_core(
                core_key,
                colors,
                shape,
                allowed,
                weights=rng.choice((-2, -1, 1, 2), size=len(allowed)),
            )

        expected_domains, expected_contradiction = enforce_generalized_arc_consistency(
            core_dict
        )
        network = ConstraintNetwork(core_dict)
        state = network.create_mac_state()
        actual = state.propagate()
        tensors = dict(state._domain_tensors)
        _assert_domain_tensors(state, tensors)

        assert actual.contradiction == expected_contradiction
        if not expected_contradiction:
            assert state.snapshot_domains() == expected_domains

            branchable = [
                color
                for color, domain in state.domains.items()
                if len(domain) > 1
            ]
            if branchable:
                color = branchable[int(rng.integers(0, len(branchable)))]
                domain = state.domains[color]
                value = domain[int(rng.integers(0, len(domain)))]
                root_domains = state.snapshot_domains()

                state.push_level()
                state.assign(color, value)
                incremental = state.propagate()
                _assert_domain_tensors(state, tensors)

                fresh_domains = dict(root_domains)
                fresh_domains[color] = (value,)
                fresh_domains, fresh_contradiction = enforce_generalized_arc_consistency(
                    core_dict,
                    domains=fresh_domains,
                )

                assert incremental.contradiction == fresh_contradiction
                if not fresh_contradiction:
                    assert state.snapshot_domains() == fresh_domains

                state.pop_level()
                assert state.snapshot_domains() == root_domains
                _assert_domain_tensors(state, tensors)

        colors = tuple(network.dimension_dict)
        solutions = [dict(zip(colors, values))
                     for values in product(*(range(network.dimension_dict[color]) for color in colors))
                     if all(core[{color: values[colors.index(color)] for color in core.colors}] != 0
                            for core in core_dict.values())]
        result = network.solve_mac()
        assert result.solved == bool(solutions)
        assert result.unsatisfiable == (not solutions)
        if result.solved:
            assert result.assignment in solutions
