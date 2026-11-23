"""Independent counting/ALS fixtures and original-entry support checks."""

from itertools import product

import numpy as np
import pytest

from demonstrations.sudoku.inferenceClusters.als_extended import (
    make_als, describe_als_xz, describe_doubly_linked_als, describe_als_chain,
    describe_death_blossom, describe_bent_subset, describe_sue_de_coq,
    describe_aligned_exclusion, detect_als_xz_extended, detect_als_chains,
    detect_als_xy_wing, detect_death_blossom, detect_bent_subsets,
    detect_wxyz_wings, detect_sue_de_coq, detect_aligned_exclusion,
)
from demonstrations.sudoku.inferenceClusters.core import RuleContext, RuleLimits, contract_cluster
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.sudoku_tests.standard_constraints import get_sudoku_constraint_network


def _fixture(name, invalid=False):
    """Return explicit finite motifs; no solved grid produces eliminations."""
    if name == "overlap":
        cells = {(0, 0): (0, 2), (0, 1): (1, 2),
                 ((2, 1) if invalid else (1, 1)): (0, 1)}
        return 2, cells, (0, 2), 2
    if name in ("double_nonrcc", "double_rcc"):
        num = 2 if name == "double_nonrcc" else 3
        cells = {(0, 0): (0, 2), (0, 1): (1, 2),
                 (((2 if num == 2 else 3), 0) if invalid else (0, 2)): (0, 1)}
        return num, cells, ((1, 0) if name == "double_nonrcc" else (0, 3)), (2 if name == "double_nonrcc" else 0)
    if name in ("chain", "blossom", "aligned_pair"):
        cells = {(0, 0): (0, 1), (0, 2): ((0, 2, 3) if invalid else (0, 2)),
                 (2, 0): (1, 2)}
        return 2, cells, (2, 2), 2
    if name == "chain_grouped":
        cells = {(0, 0): ((0, 2, 3) if invalid else (0, 3)), (0, 1): (1, 3),
                 (0, 3): (0, 2), (3, 1): (1, 2)}
        return 2, cells, (3, 3), 2
    if name == "bent":
        cells = {(0, 2): (0, 3), (2, 0): (2, 3),
                 (0, 1): ((0, 1, 2) if invalid else (0, 1)), (2, 1): (1, 2)}
        return 2, cells, (0, 0), 3
    if name == "sdc":
        cells = {(0, 0): (0, 1, 2, 3), (0, 1): (0, 1, 2, 3),
                 (0, 2): (0, 1), (1, 0): ((1, 2, 3) if invalid else (2, 3))}
        return 2, cells, (0, 3), 0
    if name == "sdc_extended":
        cells = {(0, 0): (0, 1, 2, 3), (0, 1): (0, 1, 2, 3), (0, 3): (0, 1),
                 (1, 0): ((1, 2, 4) if invalid else (2, 4)), (1, 1): (3, 4)}
        return 3, cells, (0, 4), 0
    if name == "blossom_three":
        cells = {(0, 0): (0, 1, 2), (0, 1): ((0, 3, 4) if invalid else (0, 3)),
                 (0, 2): (1, 3), (3, 0): (2, 3)}
        return 3, cells, (1, 0), 3
    if name == "aligned_triple":
        cells = {(0, 0): (0, 1), (0, 1): (1, 2), (0, 2): (0, 2, 3),
                 (1, 1): ((0, 1, 2, 4) if invalid else (0, 1, 2))}
        return 3, cells, (0, 2), 0
    raise AssertionError(name)


NAMES = ("overlap", "double_nonrcc", "double_rcc", "chain", "chain_grouped", "blossom", "bent",
         "sdc", "sdc_extended", "blossom_three", "aligned_pair", "aligned_triple")


def _context(networks, name, invalid=False, transpose=False, **limits):
    num, cells, target, digit = _fixture(name, invalid=invalid)
    net = networks[num]
    seed = RuleContext(net, num=num)
    transform = lambda cell: cell[::-1] if transpose else cell
    domains = {seed.pos(*transform(cell)): values for cell, values in cells.items()}
    return RuleContext(net, domains, num=num, limits=RuleLimits(**limits)), transform(target), digit


@pytest.fixture(scope="module")
def networks():
    return {num: ConstraintNetwork(get_sudoku_constraint_network(num=num)) for num in (2, 3)}


def _describe(ctx, name, transpose=False):
    t = lambda cell: cell[::-1] if transpose else cell
    cells = lambda selected: tuple(t(cell) for cell in selected)
    _, _, target, digit = _fixture(name)
    target = t(target)
    if name == "overlap":
        return describe_als_xz(ctx, cells(((0, 0), (0, 1))), cells(((0, 1), (1, 1))), 0, 2, target)
    if name.startswith("double_"):
        return describe_doubly_linked_als(ctx, cells(((0, 0), (0, 1))), cells(((0, 2),)),
                                         0, 1, digit, target)
    if name == "chain":
        return describe_als_chain(ctx, [cells(((0, 2),)), cells(((0, 0),)), cells(((2, 0),))],
                                  (0, 1), 2, target)
    if name == "chain_grouped":
        return describe_als_chain(ctx, [cells(((0, 3),)), cells(((0, 0), (0, 1))), cells(((3, 1),))],
                                  (0, 1), 2, target)
    if name in ("blossom", "blossom_three"):
        petals = {0: cells(((0, 2),)), 1: cells(((2, 0),))}
        if name == "blossom_three":
            petals = {0: cells(((0, 1),)), 1: cells(((0, 2),)), 2: cells(((3, 0),))}
        return describe_death_blossom(ctx, t((0, 0)), petals, digit, target)
    if name == "bent":
        return describe_bent_subset(ctx, cells(((0, 2), (2, 0), (0, 1), (2, 1))), digit, target)
    if name in ("sdc", "sdc_extended"):
        side = ((1, 0),) if name == "sdc" else ((1, 0), (1, 1))
        line_cell = (0, 2) if name == "sdc" else (0, 3)
        return describe_sue_de_coq(ctx, cells(((0, 0), (0, 1))), cells(side), cells((line_cell,)),
                                   ("col" if transpose else "row", 0), 0, digit, target)
    if name == "aligned_pair":
        return describe_aligned_exclusion(ctx, cells(((0, 0), (2, 2))),
                                         [cells(((0, 2),)), cells(((2, 0),))])
    if name == "aligned_triple":
        return describe_aligned_exclusion(ctx, cells(((0, 0), (0, 1), (0, 2))), [cells(((1, 1),))])
    raise AssertionError(name)


def _exact_support(ctx, cluster):
    """Enumerate local positions, then check selected original tensor entries.

    Every selected atom has a position link, so its Boolean truth is fixed by
    that link.  House variables can then be checked independently.  This is an
    independent finite support oracle, not a call to a solver/contractor.
    """
    by_atom, by_house = {}, {}
    positions = sorted(c for c in cluster.participating_variables if c.startswith("pos_"))
    for key in cluster.core_keys:
        core = ctx.cores[key]
        atom = next(c for c in core.colors if c.startswith("a_"))
        cat = next(c for c in core.colors if c != atom)
        if cat.startswith("pos_"):
            by_atom.setdefault(atom, []).append((cat, core))
        else:
            by_house.setdefault(cat, []).append((atom, core))
    supported = {output: set() for output in cluster.open_colors}
    for values in product(*(ctx.domains[position] for position in positions)):
        assignment = dict(zip(positions, values))
        truth = {}
        for atom, links in by_atom.items():
            allowed = [value for value in ctx.domains[atom]
                       if all(core[{atom: value, cat: assignment[cat]}] != 0 for cat, core in links)]
            assert len(allowed) <= 1
            if not allowed:
                break
            truth[atom] = allowed[0]
        else:
            if all(any(all(core[{cat: value, atom: truth[atom]}] != 0 for atom, core in links)
                       for value in ctx.domains[cat]) for cat, links in by_house.items()):
                for output in supported:
                    supported[output].add(assignment[output])
    return {output: tuple(sorted(values)) for output, values in supported.items()}


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("transpose", (False, True))
def test_compound_motifs_add_power_beyond_gac_and_equal_exact_original_support(networks, name, transpose):
    ctx, target, digit = _context(networks, name, transpose=transpose)
    before = dict(ctx.domains)
    ordinary = networks[ctx.num].create_mac_state(ctx.domains)
    assert ordinary.propagate() and digit in ordinary.domains[ctx.pos(*target)]
    cluster = _describe(ctx, name, transpose)
    result = contract_cluster(ctx, cluster)
    assert result.status == "executed" and not result.contradiction
    assert digit not in result.supported_domains[ctx.pos(*target)]
    assert result.supported_domains == _exact_support(ctx, cluster)
    assert dict(ctx.domains) == before
    assert set(cluster.core_keys) <= networks[ctx.num].coresDict.keys()


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("transpose", (False, True))
def test_invalid_near_patterns_keep_target_support_in_same_original_cluster(networks, name, transpose):
    valid, target, digit = _context(networks, name, transpose=transpose)
    cluster = _describe(valid, name, transpose)
    invalid, _, _ = _context(networks, name, invalid=True, transpose=transpose)
    result = contract_cluster(invalid, cluster)
    assert result.status == "executed" and not result.contradiction
    assert digit in result.supported_domains[invalid.pos(*target)]
    assert result.supported_domains == _exact_support(invalid, cluster)


def test_overlap_rcc_must_not_occur_in_shared_cells(networks):
    seed = RuleContext(networks[2], num=2)
    ctx = RuleContext(networks[2], {seed.pos(0, 0): (0, 2), seed.pos(0, 1): (0, 1, 2),
                                  seed.pos(1, 1): (0, 1)}, num=2)
    with pytest.raises(ValueError, match="RCC"):
        describe_als_xz(ctx, ((0, 0), (0, 1)), ((0, 1), (1, 1)), 0, 2, (0, 2))


def test_doubly_linked_als_can_project_candidate_inside_the_other_als(networks):
    # A row ALS and a column ALS have RCCs 0 (column) and 1 (box).
    # The target is a member of the column ALS, seeing every 2 in the row ALS.
    seed = RuleContext(networks[3], num=3)
    domains = {seed.pos(0, 0): (1, 2), seed.pos(0, 1): (0, 2),
               seed.pos(3, 1): (0, 2, 3), seed.pos(2, 1): (1, 3), seed.pos(1, 1): (2, 3)}
    ctx = RuleContext(networks[3], domains, num=3)
    ordinary = networks[3].create_mac_state(domains)
    assert ordinary.propagate() and 2 in ordinary.domains[seed.pos(1, 1)]
    cluster = describe_doubly_linked_als(ctx, ((0, 0), (0, 1)), ((3, 1), (2, 1), (1, 1)),
                                        0, 1, 2, (1, 1))
    result = contract_cluster(ctx, cluster)
    assert result.status == "executed" and not result.contradiction
    assert result.supported_domains[seed.pos(1, 1)] == (3,)
    assert result.supported_domains == _exact_support(ctx, cluster)


def test_death_blossom_rejects_missing_stem_alternative_and_invisible_common_digit(networks):
    ctx, _, _ = _context(networks, "blossom_three")
    petals = {0: ((0, 1),), 1: ((0, 2),), 2: ((3, 0),)}
    with pytest.raises(ValueError, match="alternative"):
        describe_death_blossom(ctx, (0, 0), {0: petals[0], 1: petals[1]}, 3, (1, 0))
    with pytest.raises(ValueError, match="target"):
        describe_death_blossom(ctx, (0, 0), petals, 3, (3, 3))


@pytest.mark.parametrize("transpose", (False, True))
def test_death_blossom_allows_overlapping_petals_and_requires_all_common_occurrences(networks, transpose):
    seed = RuleContext(networks[2], num=2)
    t = lambda cell: cell[::-1] if transpose else cell
    domains = {seed.pos(*t((0, 0))): (0, 1), seed.pos(*t((0, 2))): (0, 2),
               seed.pos(*t((0, 1))): (2, 3), seed.pos(*t((1, 1))): (1, 2)}
    petals = {0: (t((0, 2)), t((0, 1))), 1: (t((0, 1)), t((1, 1)))}
    ctx = RuleContext(networks[2], domains, num=2)
    target = t((2, 1))
    cluster = describe_death_blossom(ctx, t((0, 0)), petals, 3, target)
    ordinary = networks[2].create_mac_state(domains)
    assert ordinary.propagate() and 3 in ordinary.domains[seed.pos(*target)]
    result = contract_cluster(ctx, cluster)
    assert result.status == "executed" and 3 not in result.supported_domains[seed.pos(*target)]
    assert result.supported_domains == _exact_support(ctx, cluster)
    changed = {**domains, seed.pos(*t((0, 2))): (0, 2, 3)}
    invalid = RuleContext(networks[2], changed, num=2)
    with pytest.raises(ValueError, match="target"):
        describe_death_blossom(invalid, t((0, 0)), petals, 3, target)
    assert 3 in contract_cluster(invalid, cluster).supported_domains[seed.pos(*target)]


@pytest.mark.parametrize("transpose", (False, True))
def test_generalized_sue_de_coq_allows_same_external_digit_in_both_sectors(networks, transpose):
    seed = RuleContext(networks[3], num=3)
    t = lambda cell: cell[::-1] if transpose else cell
    values = {(0, 0): (0, 1, 2, 3), (0, 1): (0, 1, 2, 3),
              (0, 3): (0, 4), (0, 4): (1, 4), (1, 0): (2, 4), (1, 1): (3, 4)}
    ctx = RuleContext(networks[3], {seed.pos(*t(cell)): digits for cell, digits in values.items()}, num=3)
    intersection = tuple(t(cell) for cell in ((0, 0), (0, 1)))
    bs, rs = tuple(t(cell) for cell in ((1, 0), (1, 1))), tuple(t(cell) for cell in ((0, 3), (0, 4)))
    target = t((0, 5))
    cluster = describe_sue_de_coq(ctx, intersection, bs, rs,
                                ("col" if transpose else "row", 0), 0, 4, target)
    result = contract_cluster(ctx, cluster)
    assert result.status == "executed" and not result.contradiction
    assert 4 not in result.supported_domains[seed.pos(*target)]
    assert result.supported_domains == _exact_support(ctx, cluster)


def test_als_chain_validates_adjacent_restrictions_and_finite_explicit_inputs(networks):
    ctx, _, _ = _context(networks, "chain")
    with pytest.raises(ValueError, match="different"):
        describe_als_chain(ctx, [((0, 2),), ((0, 0),), ((2, 0),)], (0, 0), 2, (2, 2))
    with pytest.raises(TypeError, match="finite"):
        make_als(ctx, iter(((0, 0),)))
    with pytest.raises(ValueError, match="unsolved"):
        fixed = RuleContext(networks[2], {ctx.pos(0, 0): (0,)}, num=2)
        make_als(fixed, ((0, 0),))


AUTO = {
    "overlap": (lambda c: detect_als_xz_extended(c, kind="singly"), {"max_als_size": 2}),
    "double_nonrcc": (lambda c: detect_als_xz_extended(c, kind="doubly"), {"max_als_size": 2}),
    "chain": (detect_als_xy_wing, {"max_als_size": 1, "max_chain_length": 3}),
    "chain_grouped": (detect_als_xy_wing, {"max_als_size": 2, "max_chain_length": 3}),
    "blossom": (detect_death_blossom, {"max_als_size": 1}),
    "bent": (detect_wxyz_wings, {}),
    "sdc": (lambda c: detect_sue_de_coq(c, extended=False), {}),
    "sdc_extended": (detect_sue_de_coq, {"max_als_size": 2}),
    "blossom_three": (detect_death_blossom, {"max_als_size": 1}),
    "aligned_pair": (detect_aligned_exclusion, {"max_als_size": 1}),
    "aligned_triple": (lambda c: detect_aligned_exclusion(c, size=3), {"max_als_size": 1}),
}


@pytest.mark.parametrize("name", AUTO)
@pytest.mark.parametrize("transpose", (False, True))
def test_automatic_selectors_find_independently_specified_consequences(networks, name, transpose):
    detector, limits = AUTO[name]
    ctx, target, digit = _context(networks, name, transpose=transpose, **limits)
    detection = detector(ctx)
    matching = [cluster for cluster in detection.clusters if ctx.pos(*target) in cluster.open_colors]
    assert matching
    assert any(digit not in contract_cluster(ctx, cluster).supported_domains[ctx.pos(*target)]
               for cluster in matching)
    assert detection.steps <= ctx.limits.max_steps


@pytest.mark.parametrize("name", AUTO)
@pytest.mark.parametrize("transpose", (False, True))
def test_automatic_selectors_do_not_remove_invalid_near_pattern_target(networks, name, transpose):
    detector, limits = AUTO[name]
    ctx, target, digit = _context(networks, name, invalid=True, transpose=transpose, **limits)
    for cluster in detector(ctx).clusters:
        if ctx.pos(*target) in cluster.open_colors:
            result = contract_cluster(ctx, cluster)
            assert result.status == "executed"
            assert digit in result.supported_domains[ctx.pos(*target)]


@pytest.mark.parametrize("transpose", (False, True))
def test_four_als_chain_and_endpoint_digit_restrictions(networks, transpose):
    seed = RuleContext(networks[2], num=2)
    t = lambda cell: cell[::-1] if transpose else cell
    cells = ((0, 0), (0, 2), (2, 2), (2, 0))
    digits = ((0, 1), (1, 2), (2, 3), (0, 3))
    domains = {seed.pos(*t(cell)): options for cell, options in zip(cells, digits)}
    ctx = RuleContext(networks[2], domains, num=2, limits=RuleLimits(max_als_size=1, max_chain_length=4))
    sets = [ (t(cell),) for cell in cells ]
    target = t((1, 0))
    cluster = describe_als_chain(ctx, sets, (1, 2, 3), 0, target)
    result = contract_cluster(ctx, cluster)
    assert 0 not in result.supported_domains[ctx.pos(*target)]
    assert result.supported_domains == _exact_support(ctx, cluster)
    matches = [cluster for cluster in detect_als_chains(ctx, lengths=(4,)).clusters
               if ctx.pos(*target) in cluster.open_colors]
    assert any(0 not in contract_cluster(ctx, cluster).supported_domains[ctx.pos(*target)] for cluster in matches)
    with pytest.raises(ValueError, match="endpoint"):
        describe_als_chain(ctx, sets, (1, 2, 3), 1, target)


@pytest.mark.parametrize("detector", (detect_als_xz_extended, detect_als_chains, detect_als_xy_wing,
                                     detect_death_blossom, detect_bent_subsets, detect_wxyz_wings,
                                     detect_sue_de_coq, detect_aligned_exclusion))
def test_every_automatic_scan_obeys_work_limit_before_patterns(networks, detector):
    ctx = RuleContext(networks[2], num=2, limits=RuleLimits(max_steps=5))
    result = detector(ctx)
    assert result.steps == 5 and result.truncated and not result.clusters


def test_catalogue_pattern_and_execution_caps_are_distinct(networks):
    ctx, _, _ = _context(networks, "overlap", max_als_sets=1)
    result = detect_als_xz_extended(ctx)
    assert result.truncated and not result.clusters
    ctx, _, _ = _context(networks, "bent", max_patterns=1, max_work=1)
    result = detect_bent_subsets(ctx)
    assert len(result.clusters) == 1 and result.truncated
    assert contract_cluster(ctx, result.clusters[0]).status == "skipped_budget"


def test_large_configured_lengths_do_not_materialize_large_ranges(networks):
    ctx = RuleContext(networks[2], num=2,
                      limits=RuleLimits(max_chain_length=10**12, max_subset_size=10**12,
                                        max_als_sets=10**12, max_steps=5))
    assert detect_als_chains(ctx).steps == 5
    assert detect_bent_subsets(ctx).steps == 5
    # More distinct base cells than exist on the board is empty immediately.
    empty = detect_aligned_exclusion(ctx, size=10**12)
    assert not empty.clusters and empty.steps == 0
    fixture, _, _ = _context(networks, "chain")
    with pytest.raises(ValueError, match="explicit finite"):
        make_als(fixture, ((0, 0),), house=iter(("row", 0)))


def test_different_domain_snapshots_and_generic_backtracking_remain_independent(networks):
    valid, target, digit = _context(networks, "bent")
    cluster = _describe(valid, "bent")
    net = networks[2]
    state = net.create_mac_state()
    assert state.propagate()
    before = state.snapshot_domains()
    state.push_level()
    for color, values in valid.domains.items():
        if values != tuple(range(valid.dimensions[color])):
            state.restrict_domain(color, values)
    assert state.propagate()
    result = contract_cluster(RuleContext(net, state.domains, num=2), cluster)
    state.add_cores(result.summaries, reversible=True)
    assert state.propagate() and digit not in state.domains[valid.pos(*target)]
    state.pop_level()
    assert state.snapshot_domains() == before
    invalid, _, _ = _context(networks, "bent", invalid=True)
    assert digit in contract_cluster(invalid, cluster).supported_domains[invalid.pos(*target)]


def test_signed_original_values_and_source_arrays_are_preserved(networks):
    ctx, target, digit = _context(networks, "chain")
    cluster = _describe(ctx, "chain")
    cores = {key: core.clone() for key, core in networks[2].coresDict.items()}
    for core in cores.values():
        core.values *= np.where(np.indices(core.shape).sum(axis=0) % 2, -2, 3)
    before = {key: core.values.copy() for key, core in cores.items()}
    signed = RuleContext(cores, ctx.domains, num=2)
    result = contract_cluster(signed, cluster)
    assert digit not in result.supported_domains[signed.pos(*target)]
    assert result.supported_domains == _exact_support(signed, cluster)
    for key, values in before.items():
        np.testing.assert_array_equal(cores[key].values, values)
