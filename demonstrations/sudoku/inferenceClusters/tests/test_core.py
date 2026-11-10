"""Independent support checks and caller-managed reversible application."""
from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from demonstrations.sudoku.inferenceClusters.core import (
    RuleContext, RuleLimits, contract_cluster,
)
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.sudoku_tests.standard_constraints import get_sudoku_constraint_network
from tnreason import engine


def network():
    return ConstraintNetwork(get_sudoku_constraint_network(num=2))


def pair(context, **kwargs):
    return context.describe("test_pair", (context.pos(0, 0), context.pos(0, 2),
                            context.unit("row", 0, 0), context.unit("row", 0, 1)), **kwargs)


def exact_projection(context, cluster, use_domains):
    """Exhaust all small selected assignments, independently of contraction."""
    colors = cluster.participating_variables
    indices = [context.domains[c] if use_domains else range(context.dimensions[c]) for c in colors]
    result = np.zeros(tuple(context.dimensions[c] for c in cluster.open_colors), dtype=bool)
    for values in product(*indices):
        assignment = dict(zip(colors, values))
        if all(context.cores[k][{c: assignment[c] for c in context.cores[k].colors}] != 0
               for k in cluster.core_keys):
            result[tuple(assignment[c] for c in cluster.open_colors)] = True
    return result


@pytest.mark.parametrize("conditioned", [False, True])
@pytest.mark.parametrize("perturbed", [False, True])
def test_joint_and_unary_support_equal_independent_enumeration(conditioned, perturbed):
    net = network()
    if perturbed:
        for core in net.coresDict.values():
            core.values *= np.where(np.indices(core.shape).sum(axis=0) % 2, -2, 3)
        net.coresDict["pos_0_0_0_0_a_0_0_0_0_0"].values[0, 0] = 0
    ctx = RuleContext(net, {"pos_0_0_0_0": (0, 1), "pos_0_0_1_0": (0, 1)}, num=2)
    description = pair(ctx)
    expected = exact_projection(ctx, description, conditioned)
    joint = contract_cluster(ctx, description, joint=True, use_domains=conditioned)
    assert joint.status == "executed"
    np.testing.assert_array_equal(next(iter(joint.summaries.values())).values, expected)
    result = contract_cluster(ctx, description, use_domains=conditioned)
    for axis, color in enumerate(description.open_colors):
        projection = np.any(expected, axis=tuple(i for i in range(expected.ndim) if i != axis))
        assert result.supported_domains[color] == tuple(np.flatnonzero(projection))
    assert result.contradiction == (not expected.any())


def test_sparse_realized_support_and_iterator_state_are_preserved():
    net = network()
    domains = {"pos_0_0_0_0": (0, 1), "pos_0_0_1_0": (0, 1)}
    ctx = RuleContext(net, domains, num=2)
    expected = contract_cluster(ctx, pair(ctx), joint=True)
    key = "pos_0_0_0_0_a_0_0_0_0_0"
    original = net.coresDict[key]
    cancelled = {"pos_0_0_0_0": 0, "a_0_0_0_0_0": 0}
    slices = [(value, assignment) for value, assignment in original if value != 0]
    slices += [(5, cancelled), (-5, cancelled)]
    sparse = engine.create_from_slice_iterator(original.shape, original.colors, slices,
                                               coreType="PolynomialCore")
    net.coresDict[key] = sparse
    # A different original NumpyCore's consumed iterator is irrelevant to reads.
    another = net.coresDict["row_0_0_0_a_0_0_0_0_0"]
    next(another)
    index = another.index
    ctx = RuleContext(net, domains, num=2)
    actual = contract_cluster(ctx, pair(ctx), joint=True)
    np.testing.assert_array_equal(next(iter(expected.summaries.values())).values,
                                  next(iter(actual.summaries.values())).values)
    assert sparse.values == slices and sparse[cancelled] == 0
    assert another.index == index


def test_domains_snapshot_and_constraint_sensitive_bounded_cache():
    net = network()
    domains = {"pos_0_0_0_0": [0, 1], "pos_0_0_1_0": [0, 1]}
    ctx = RuleContext(net, domains, num=2, limits=RuleLimits(max_cache_entries=2))
    desc = pair(ctx)
    before = contract_cluster(ctx, desc)
    assert before.supported_domains["row_0_0_0"] == (0, 2)
    domains["pos_0_0_0_0"].append(2)
    assert ctx.domains["pos_0_0_0_0"] == (0, 1)
    assert len(ctx._support_cache) <= 2
    net.coresDict[desc.core_keys[0]].values.fill(0)
    after = contract_cluster(ctx, desc)
    assert after.contradiction
    assert len(ctx._support_cache) <= 2
    fresh = RuleContext(network(), domains, num=2)
    assert contract_cluster(fresh, pair(fresh)).supported_domains["row_0_0_0"] == (0, 1, 2, 3)


@pytest.mark.parametrize("limit,kwargs", [
    ("max_cores", {}), ("max_work", {}), ("max_size", {}),
    ("max_planning_steps", {}),
    ("max_output_size", {"joint": True}),
])
def test_recognized_description_has_explicit_budget_skip(limit, kwargs):
    ctx = RuleContext(network(), num=2, limits=replace(RuleLimits(), **{limit: 1}))
    desc = pair(ctx)
    result = contract_cluster(ctx, desc, **kwargs)
    assert result.status == "skipped_budget" and result.reason
    assert not result.summaries and not result.supported_domains and not result.contradiction


def test_empty_domains_and_scalar_contradictions():
    ctx = RuleContext(network(), {"pos_0_0_0_0": ()}, num=2)
    result = contract_cluster(ctx, pair(ctx))
    assert result.status == "executed" and result.contradiction
    assert all(values == () for values in result.supported_domains.values())
    ctx = RuleContext(network(), num=2)
    result = contract_cluster(ctx, pair(ctx, outputs=()))
    assert not result.contradiction and next(iter(result.summaries.values())).values.shape == ()


def test_unencoded_assumptions_are_rejected():
    ctx = RuleContext(network(), num=2)
    with pytest.raises(ValueError, match="Unencoded extra assumptions"):
        contract_cluster(ctx, pair(ctx, assumptions=("unique_solution",)))


def test_domain_summaries_are_reversible_and_sibling_contexts_are_independent():
    net = network()
    saved = {k: c.values.copy() for k, c in net.coresDict.items()}
    state = net.create_mac_state()
    assert state.propagate()
    root = state.snapshot_domains()
    keys = set(state._cores)
    for digits in ((0, 1), (2, 3)):
        state.push_level()
        state.restrict_domain("pos_0_0_0_0", digits)
        state.restrict_domain("pos_0_0_1_0", digits)
        assert state.propagate()
        # Ordinary binary-link GAC cannot find the pair exclusion.
        assert state.domains["pos_0_0_0_1"] == (0, 1, 2, 3)
        ctx = RuleContext(net, state.domains, num=2)
        categories = (ctx.pos(0, 0), ctx.pos(0, 2), *(ctx.unit("row", 0, d) for d in digits))
        result = contract_cluster(ctx, ctx.describe("test_pair", categories))
        state.add_cores(result.summaries, reversible=True)
        assert state.propagate()
        assert state.domains["pos_0_0_0_1"] == tuple(d for d in range(4) if d not in digits)
        state.pop_level()
        assert state.snapshot_domains() == root and set(state._cores) == keys
    for key, core in net.coresDict.items():
        np.testing.assert_array_equal(core.values, saved[key])


def test_domain_free_summary_preserves_witness_in_another_domain_snapshot():
    net = network()
    ctx = RuleContext(net, {"pos_0_0_0_0": (0, 1), "pos_0_0_1_0": (0, 1)}, num=2)
    result = contract_cluster(ctx, pair(ctx), joint=True, use_domains=False)
    state = net.create_mac_state({"pos_0_0_0_0": (2,), "pos_0_0_1_0": (3,)})
    state.add_cores(result.summaries)
    assert state.propagate()


def test_local_indicator_domains_participate_and_domain_inputs_are_explicit():
    ctx = RuleContext(network(), {"a_0_0_0_0_0": (1,)}, num=2)
    assert ctx.options(ctx.pos(0, 0)) == frozenset((0,))
    desc = ctx.describe("single", (ctx.pos(0, 0),), atoms=ctx.categories[ctx.pos(0, 0)],
                        outputs=(ctx.pos(0, 0),))
    assert set(desc.domain_inputs) == set(desc.participating_variables)
    assert set(desc.internal_variables) == set(desc.participating_variables) - set(desc.open_colors)
    assert contract_cluster(ctx, desc).supported_domains[ctx.pos(0, 0)] == (0,)


def test_limits_validation_and_missing_links():
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            RuleLimits(max_steps=value)
    with pytest.raises(ValueError, match="max_board_size"):
        RuleContext({}, num=10)
    net = network()
    del net.coresDict["pos_0_0_0_0_a_0_0_0_0_0"]
    ctx = RuleContext(net, num=2)
    with pytest.raises(ValueError, match="Missing Sudoku link"):
        pair(ctx)


def test_scan_budget_is_explicit():
    ctx = RuleContext(network(), num=2, limits=RuleLimits(max_steps=1))
    scan = ctx.scan()
    assert scan.step() and not scan.step()
    assert scan.finish(()).truncated and scan.finish(()).steps == 1


@pytest.mark.parametrize("atom_domain", [(0, 1), (0,), (1,)])
@pytest.mark.parametrize("conditioned", [False, True])
def test_higher_degree_indicators_and_fixed_outputs_equal_exact_support(atom_domain, conditioned):
    net = network()
    domains = {"a_0_0_0_0_0": atom_domain, "row_0_0_0": (0,),
               "pos_0_0_0_0": (0, 1), "col_0_0_0": (0, 1)}
    ctx = RuleContext(net, domains, num=2)
    description = ctx.describe("degree_four", (ctx.pos(0, 0), ctx.unit("row", 0, 0),
                               ctx.unit("col", 0, 0), ctx.unit("square", 0, 0)),
                               outputs=(ctx.pos(0, 0), ctx.unit("row", 0, 0)))
    # Arbitrary realized signed support ensures execution uses the original
    # tensors instead of assuming the standard categorical truth table.
    rng = np.random.default_rng(12)
    for key in description.core_keys:
        net.coresDict[key].values[:] = rng.integers(-1, 2, size=net.coresDict[key].shape)
    expected = exact_projection(ctx, description, conditioned)
    result = contract_cluster(ctx, description, joint=True, use_domains=conditioned)
    assert result.status == "executed"
    np.testing.assert_array_equal(next(iter(result.summaries.values())).values, expected)
