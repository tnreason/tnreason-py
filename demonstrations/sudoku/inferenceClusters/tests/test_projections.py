"""Original-support proofs for templates, overlays and finite forcing nets."""
from itertools import permutations

import numpy as np
import pytest

from demonstrations.sudoku.inferenceClusters.core import RuleContext, RuleLimits, contract_cluster
from demonstrations.sudoku.inferenceClusters.projections import (
    combine_clusters, describe_forcing_net, describe_pattern_overlay, describe_templates,
    describe_complete_support, detect_complete_support, detect_proof_regions,
    detect_nishio, detect_pattern_overlay, detect_templates,
)
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.sudoku_tests.standard_constraints import get_sudoku_constraint_network


@pytest.fixture(scope="module")
def network():
    return ConstraintNetwork(get_sudoku_constraint_network(num=2))


def closed(network, domains, limits=None):
    state = network.create_mac_state(domains)
    assert state.propagate()
    return state, RuleContext(network, state.domains, num=2, limits=limits)


def template_oracle(ctx, digit):
    """Independent 4x4 template enumeration, used only as a test oracle."""
    valid = []
    for cols in permutations(range(ctx.n)):
        if len({r // ctx.num * ctx.num + c // ctx.num for r, c in enumerate(cols)}) != ctx.n:
            continue
        assignment = {}
        for r, c in enumerate(cols):
            assignment[ctx.unit("row", r, digit)] = c
            assignment[ctx.unit("col", c, digit)] = r
            box = r // ctx.num * ctx.num + c // ctx.num
            assignment[ctx.unit("square", box, digit)] = r % ctx.num * ctx.num + c % ctx.num
        for r in range(ctx.n):
            for c in range(ctx.n):
                assignment[ctx.atom(r, c, digit)] = int(cols[r] == c)
        if any(value not in ctx.domains[color] for color, value in assignment.items()):
            continue
        cluster = describe_templates(ctx, digit)
        if all(ctx.cores[k][{c: assignment[c] for c in ctx.cores[k].colors}] != 0 for k in cluster.core_keys):
            valid.append(cols)
    return {ctx.unit("row", r, digit): tuple(sorted({cols[r] for cols in valid})) for r in range(ctx.n)}


@pytest.mark.parametrize("transpose", [False, True])
def test_templates_and_nishio_add_power_and_match_independent_support(network, transpose):
    naming = RuleContext(network, num=2)
    kind = "col" if transpose else "row"
    domains = {naming.unit(kind, r, 0): (0, 2) for r in (0, 2)}
    state, ctx = closed(network, domains)
    assert state.domains[naming.unit(kind, 1, 0)] == (0, 1, 2, 3)
    for detection in (detect_templates(ctx, digits=(0,)), detect_nishio(ctx, digits=(0,))):
        assert not detection.truncated and len(detection.clusters) == 1
        result = contract_cluster(ctx, detection.clusters[0])
        assert result.status == "executed" and not result.contradiction
        assert result.supported_domains == template_oracle(ctx, 0)
        assert any(set(ctx.domains[c]) - set(d) for c, d in result.supported_domains.items())
    near = dict(domains)
    near[naming.unit(kind, 2, 0)] = (0, 1, 2)
    _, changed = closed(network, near)
    result = contract_cluster(changed, describe_templates(changed, 0))
    assert result.supported_domains == template_oracle(changed, 0)
    assert 0 in result.supported_domains[naming.unit("row", 1 if not transpose else 0, 0)]


@pytest.mark.parametrize("transpose", [False, True])
def test_pattern_overlay_couples_digits_beyond_single_digit_templates(network, transpose):
    naming = RuleContext(network, num=2)
    cells = ((0, 0), (2, 0)) if transpose else ((0, 0), (0, 2))
    target = naming.unit("col" if transpose else "row", 0, 0)
    domains = {naming.pos(*c): (0, 1) for c in cells}
    state, ctx = closed(network, domains)
    assert state.domains[target] == tuple(range(4))
    single = contract_cluster(ctx, describe_templates(ctx, 0, outputs=(target,)))
    assert single.status == "executed" and single.supported_domains[target] == tuple(range(4))
    selected = detect_pattern_overlay(ctx, digits=(0, 1), outputs=(target,))
    result = contract_cluster(ctx, selected.clusters[0])
    assert result.status == "executed" and result.supported_domains[target] == (0, 2)
    # Breaking the pair allows a concrete target placement again.
    domains[naming.pos(*cells[1])] = (0, 1, 2)
    _, changed = closed(network, domains)
    near = contract_cluster(changed, describe_pattern_overlay(changed, (0, 1), outputs=(target,)))
    assert near.status == "executed" and 1 in near.supported_domains[target]


def test_explicit_exhaustive_forcing_proof_union_is_exact_and_branch_safe(network):
    naming = RuleContext(network, num=2)
    domains = {naming.pos(0, 0): (0, 1), naming.pos(0, 2): (0, 1)}
    state, ctx = closed(network, domains)
    source, target = naming.pos(0, 0), naming.unit("row", 0, 0)
    pair = ctx.describe("pair_proof", (source, naming.pos(0, 2), target, naming.unit("row", 0, 1)))
    net = describe_forcing_net(ctx, source, {0: pair, 1: pair}, outputs=(target,))
    before = state.snapshot_domains()
    result = contract_cluster(ctx, net)
    assert result.status == "executed" and result.supported_domains[target] == (0, 2)
    expected = contract_cluster(ctx, pair)
    assert expected.supported_domains[target] == result.supported_domains[target]
    state.push_level()
    state.add_cores(result.summaries, reversible=True)
    assert state.propagate() and state.domains[naming.pos(0, 1)] == (1, 2, 3)
    state.pop_level()
    assert state.snapshot_domains() == before
    with pytest.raises(ValueError, match="exhaust"):
        describe_forcing_net(ctx, source, {0: pair}, outputs=(target,))
    # The same original proof under different domains cannot leak the result.
    _, near = closed(network, {source: (0, 1), naming.pos(0, 2): (0, 1, 2)})
    assert contract_cluster(near, net).supported_domains[target] == (0, 1, 2, 3)


@pytest.mark.parametrize("transpose", [False, True])
def test_automatic_finite_forcing_regions_have_real_support_effect(network, transpose):
    naming = RuleContext(network, num=2)
    cells = ((0, 0), (2, 0)) if transpose else ((0, 0), (0, 2))
    source = naming.unit("col" if transpose else "row", 0, 0)
    domains = {naming.pos(*c): (0, 1) for c in cells}
    state, ctx = closed(network, domains)
    detection = detect_proof_regions(ctx, seeds=(source,), radius=2)
    assert detection.clusters
    result = contract_cluster(ctx, detection.clusters[0])
    assert result.status == "executed" and result.supported_domains[source] == (0, 2)
    assert state.domains[source] == (0, 1, 2, 3)
    domains[naming.pos(*cells[1])] = (0, 1, 2)
    _, near = closed(network, domains)
    changed = contract_cluster(near, detection.clusters[0])
    assert changed.status == "executed" and 1 in changed.supported_domains[source]


def test_projection_budget_skips_and_signed_originals(network):
    ctx = RuleContext(network, num=2, limits=RuleLimits(max_steps=2))
    for detector in (detect_templates, detect_pattern_overlay, detect_proof_regions):
        result = detector(ctx)
        assert result.steps <= 2 and result.truncated and not result.clusters
    ctx = RuleContext(network, num=2, limits=RuleLimits(max_cores=1))
    assert contract_cluster(ctx, describe_templates(ctx, 0)).status == "skipped_budget"
    with pytest.raises(ValueError):
        describe_pattern_overlay(ctx, (0,))
    with pytest.raises(ValueError):
        detect_templates(ctx, digits=(True,))
    with pytest.raises(TypeError):
        detect_templates(ctx, digits=iter((0,)))
    cores = get_sudoku_constraint_network(num=2)
    for core in cores.values():
        core.values *= np.where(np.indices(core.shape).sum(axis=0) % 2, -2, 3)
    signed = RuleContext(cores, num=2)
    actual = contract_cluster(signed, describe_templates(signed, 0))
    assert actual.supported_domains == template_oracle(signed, 0)


def test_arbitrary_original_tensor_projection_has_explicit_metadata(network):
    ctx = RuleContext(network, num=2)
    original = describe_templates(ctx, 0)
    duplicate = ctx.describe_tensors("original_region", original.core_keys, original.open_colors)
    assert set(duplicate.participating_variables) == set(original.participating_variables)
    assert set(duplicate.internal_variables) == set(original.internal_variables)
    assert duplicate.domain_inputs == duplicate.participating_variables
    assert contract_cluster(ctx, duplicate).supported_domains == contract_cluster(ctx, original).supported_domains
    combined = combine_clusters(ctx, "union", (original, duplicate))
    assert set(combined.core_keys) == set(original.core_keys)
    with pytest.raises(ValueError):
        ctx.describe_tensors("missing", ("not_original",))
    with pytest.raises(TypeError):
        ctx.describe_tensors("infinite", iter(original.core_keys))


def test_templates_detect_internal_contradiction(network):
    ctx = RuleContext(network, {"row_0_0_0": (), "row_0_0_1": (0, 1)}, num=2)
    result = contract_cluster(ctx, describe_templates(ctx, 0))
    assert result.status == "executed" and result.contradiction
    assert all(values == () for values in result.supported_domains.values())


def test_complete_original_support_includes_explicit_extra_constraints(network):
    naming = RuleContext(network, num=2)
    source = naming.pos(0, 0)
    extra = type(next(iter(network.coresDict.values())))(
        values=np.array([True, False, True, True]), colors=[source])
    ctx = RuleContext({**network.coresDict, "explicit_extra": extra}, num=2)
    description = describe_complete_support(ctx, outputs=(source,))
    assert "explicit_extra" in description.core_keys
    # A complete region's execution limits are explicit, independently of
    # whether its topology was successfully described.
    tiny = RuleContext(ctx.cores, num=2, limits=RuleLimits(max_cores=1))
    assert contract_cluster(tiny, description).status == "skipped_budget"
    assert detect_complete_support(tiny).clusters


def test_complete_support_nine_by_nine_fixed_axes_execute_without_a_solution_lookup():
    num, n = 3, 9
    net = ConstraintNetwork(get_sudoku_constraint_network(num=num))
    naming = RuleContext(net, num=num)
    # Independently generated valid givens are inputs, not a solution consulted
    # by a rule. Their singleton domains exercise large-network projection.
    grid = tuple(tuple((r * num + r // num + c) % n for c in range(n)) for r in range(n))
    domains = {naming.pos(r, c): (grid[r][c],) for r in range(n) for c in range(n)}
    state = net.create_mac_state(domains)
    assert state.propagate()
    ctx = RuleContext(net, state.domains, num=num, limits=RuleLimits(max_cores=3000))
    result = contract_cluster(ctx, describe_complete_support(ctx, outputs=(naming.pos(0, 0),)))
    assert result.status == "executed" and not result.contradiction
    assert result.supported_domains[naming.pos(0, 0)] == (0,)
