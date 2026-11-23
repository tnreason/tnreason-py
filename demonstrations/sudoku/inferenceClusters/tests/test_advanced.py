"""Independent small motifs, with exact support and invalid witnesses."""

from dataclasses import replace
from itertools import product

import pytest

from demonstrations.sudoku.inferenceClusters.advanced import (
    detect_als_xz, detect_simple_coloring, detect_xy_chains,
    detect_xy_wings, detect_xyz_wings,
)
from demonstrations.sudoku.inferenceClusters.core import (
    RuleContext, RuleLimits, contract_cluster,
)
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.sudoku_tests.standard_constraints import (
    get_sudoku_constraint_network,
)


DETECTORS = {
    "xy": detect_xy_wings,
    "xyz": detect_xyz_wings,
    "chain": detect_xy_chains,
    "color": detect_simple_coloring,
    "als": detect_als_xz,
}


@pytest.fixture(scope="module")
def network():
    return ConstraintNetwork(get_sudoku_constraint_network(num=2))


def _fixture(ctx, family, invalid=False, transpose=False):
    positions, units = {}, []
    if family == "xy":
        positions = {(0, 0): (0, 1), (0, 2): (0, 2), (2, 0): (1, 2)}
        target, digit = (2, 2), 2
        if invalid:
            positions[0, 2] = (0, 2, 3)
        witness = ((0, 1, 3, 2), (2, 3, 1, 0), (1, 0, 2, 3), (3, 2, 0, 1))
    elif family == "xyz":
        positions = {(0, 0): (0, 1, 2), (0, 1): (0, 2), (2, 0): (1, 2)}
        target, digit = (1, 0), 2
        if invalid:
            positions[0, 1] = (0, 2, 3)
        witness = ((0, 3, 1, 2), (2, 1, 3, 0), (1, 0, 2, 3), (3, 2, 0, 1))
    elif family == "chain":
        positions = {(0, 0): (0, 1), (0, 2): (1, 2),
                     (2, 2): (2, 3), (2, 0): (0, 3)}
        target, digit = (1, 0), 0
        if invalid:
            positions[0, 2] = (1, 2, 3)
        witness = ((1, 2, 3, 0), (0, 3, 1, 2), (3, 0, 2, 1), (2, 1, 0, 3))
    elif family == "color":
        units = [("row", 0, 0, (0, 2)), ("col", 2, 0, (0, 2)),
                 ("row", 2, 0, (0, 1, 2) if invalid else (0, 2))]
        target, digit = (1, 0), 0
        witness = ((1, 2, 0, 3), (0, 3, 1, 2), (3, 0, 2, 1), (2, 1, 3, 0))
    else:
        positions = {(0, 0): (0, 1), (0, 1): (1, 2),
                     (2, 0): (0, 3), (2, 1): (2, 3)}
        target, digit = (1, 1), 2
        if invalid:
            # X's occurrences no longer see one another; Z remains shared.
            positions[2, 2] = positions.pop((2, 0))
        witness = ((0, 1, 2, 3), (3, 2, 1, 0), (1, 3, 0, 2), (2, 0, 3, 1))

    def transform(cell):
        return cell[::-1] if transpose else cell

    domains = {ctx.pos(*transform(cell)): options for cell, options in positions.items()}
    for kind, index, value, options in units:
        if transpose:
            kind = {"row": "col", "col": "row"}[kind]
        domains[ctx.unit(kind, index, value)] = options
    if transpose:
        witness = tuple(zip(*witness))
    return domains, ctx.pos(*transform(target)), digit, witness


def _exact_local_support(ctx, cluster, target):
    """Enumerate position assignments and check original realized entries.

    Position incidences fix most indicator values; enumerate the few remaining
    indicators explicitly. With indicators fixed, the unit variables can be
    checked independently. This uses neither the implementation under test nor
    a search solver or known completed solution to obtain conclusions.
    """
    atoms = {}
    for key in cluster.core_keys:
        core = ctx.cores[key]
        atom = next(color for color in core.colors if color.startswith("a_"))
        category = next(color for color in core.colors if color != atom)
        atoms.setdefault(atom, []).append((category, core))
    positions = sorted(color for color in cluster.participating_variables
                       if color.startswith("pos_"))
    units = sorted(color for color in cluster.participating_variables
                   if not color.startswith(("pos_", "a_")))
    result = set()
    for values in product(*(ctx.domains[color] for color in positions)):
        assignment = dict(zip(positions, values))
        allowed_truth = {}
        for atom, links in atoms.items():
            position_links = [(cat, core) for cat, core in links if cat in assignment]
            supported = [value for value in ctx.domains[atom]
                         if all(core[{atom: value, cat: assignment[cat]}] != 0
                                for cat, core in position_links)]
            if position_links:
                assert len(supported) <= 1
            if not supported:
                break
            allowed_truth[atom] = supported
        else:
            for truth_values in product(*allowed_truth.values()):
                truth = dict(zip(allowed_truth, truth_values))
                if all(any(all(core[{atom: truth[atom], cat: value}] != 0
                               for atom, links in atoms.items()
                               for cat, core in links if cat == unit)
                           for value in ctx.domains[unit]) for unit in units):
                    result.add(assignment[target])
                    break
    return tuple(sorted(result))


def _witness_satisfies_original(ctx, grid):
    assignment = {}
    for row in range(ctx.n):
        for col in range(ctx.n):
            digit = grid[row][col]
            assignment[ctx.pos(row, col)] = digit
            for value in range(ctx.n):
                assignment[ctx.atom(row, col, value)] = int(value == digit)
    for category, atoms in ctx.categories.items():
        if not category.startswith("pos_"):
            assignment[category] = next(index for index, atom in enumerate(atoms)
                                        if assignment[atom])
    assert all(value in ctx.domains[color] for color, value in assignment.items())
    assert all(core[{color: assignment[color] for color in core.colors}] != 0
               for core in ctx.cores.values())


@pytest.mark.parametrize("family", DETECTORS)
@pytest.mark.parametrize("transpose", [False, True])
def test_advanced_motifs_strengthen_gac_and_match_independent_support(network, family, transpose):
    base = RuleContext(network, num=2)
    domains, target, digit, _ = _fixture(base, family, transpose=transpose)
    original_domains = dict(domains)
    ordinary = network.create_mac_state(domains)
    assert ordinary.propagate()
    assert digit in ordinary.domains[target]
    ctx = RuleContext(network, domains, num=2)
    detection = DETECTORS[family](ctx)
    assert not detection.truncated
    cluster = next(cluster for cluster in detection.clusters if target in cluster.open_colors)
    result = contract_cluster(ctx, cluster)
    assert result.status == "executed" and not result.contradiction
    assert digit not in result.supported_domains[target]
    assert result.supported_domains[target] == _exact_local_support(ctx, cluster, target)
    assert domains == original_domains
    assert set(cluster.core_keys) <= network.coresDict.keys()


@pytest.mark.parametrize("family", DETECTORS)
@pytest.mark.parametrize("transpose", [False, True])
def test_invalid_near_patterns_preserve_explicit_global_witness(network, family, transpose):
    base = RuleContext(network, num=2)
    domains, target, digit, witness = _fixture(base, family, invalid=True, transpose=transpose)
    ctx = RuleContext(network, domains, num=2)
    _witness_satisfies_original(ctx, witness)
    # These independently constructed complete witnesses certify that the
    # candidate is possible. They are not used to obtain any elimination.
    for cluster in DETECTORS[family](ctx).clusters:
        if target in cluster.open_colors:
            result = contract_cluster(ctx, cluster)
            assert result.status == "executed"
            assert digit in result.supported_domains[target]


@pytest.mark.parametrize("transpose", [False, True])
def test_coloring_wrap_contracts_same_color_conflict(network, transpose):
    base = RuleContext(network, num=2)
    units = [("row", 0), ("col", 1)]
    if transpose:
        units = [("col" if kind == "row" else "row", index) for kind, index in units]
    domains = {base.unit(kind, index, 0): (0, 1) for kind, index in units}
    ctx = RuleContext(network, domains, num=2)
    wraps = [cluster for cluster in detect_simple_coloring(ctx).clusters
             if "color wrap" in cluster.evidence[0]]
    assert wraps
    for cluster in wraps:
        result = contract_cluster(ctx, cluster)
        assert result.status == "executed"
        for target in cluster.open_colors:
            assert 0 not in result.supported_domains[target]
            assert result.supported_domains[target] == _exact_local_support(ctx, cluster, target)
    # Widen both conjugate pairs: the odd conjugate path is absent.
    invalid = {unit: (0, 1, 2) for unit in domains}
    invalid_ctx = RuleContext(network, invalid, num=2)
    assert not detect_simple_coloring(invalid_ctx).clusters


def test_xyz_target_must_see_pivot_as_well_as_both_pincers(network):
    base = RuleContext(network, num=2)
    domains = {base.pos(0, 0): (0, 1, 2), base.pos(0, 2): (0, 2),
               base.pos(2, 0): (1, 2)}
    ctx = RuleContext(network, domains, num=2)
    # (2,2) sees the two pincers through row/column, but not the pivot.
    assert all(base.pos(2, 2) not in cluster.open_colors
               for cluster in detect_xyz_wings(ctx).clusters)


@pytest.mark.parametrize("family", DETECTORS)
def test_detection_work_is_bounded_even_before_any_pattern(network, family):
    ctx = RuleContext(network, num=2, limits=RuleLimits(max_steps=5))
    detection = DETECTORS[family](ctx)
    assert detection.steps == 5 and detection.truncated
    assert not detection.clusters


@pytest.mark.parametrize("family", ["chain", "color", "als"])
def test_structural_bounds_limit_advanced_coverage(network, family):
    base = RuleContext(network, num=2)
    domains, target, digit, _ = _fixture(base, family)
    limits = replace(RuleLimits(), max_als_size=1, max_chain_length=3)
    ctx = RuleContext(network, domains, num=2, limits=limits)
    for cluster in DETECTORS[family](ctx).clusters:
        if target in cluster.open_colors:
            assert digit in contract_cluster(ctx, cluster).supported_domains[target]


def test_pattern_cap_is_separate_from_execution_budget(network):
    base = RuleContext(network, num=2)
    domains, _, _, _ = _fixture(base, "chain")
    ctx = RuleContext(network, domains, num=2,
                      limits=RuleLimits(max_patterns=1, max_size=1))
    detection = detect_xy_chains(ctx)
    assert len(detection.clusters) == 1 and detection.truncated
    assert contract_cluster(ctx, detection.clusters[0]).status == "skipped_budget"


def test_domain_changes_do_not_reuse_a_previous_xy_wing_conclusion(network):
    base = RuleContext(network, num=2)
    domains, target, digit, _ = _fixture(base, "xy")
    first = RuleContext(network, domains, num=2)
    cluster = next(cluster for cluster in detect_xy_wings(first).clusters
                   if target in cluster.open_colors)
    assert digit not in contract_cluster(first, cluster).supported_domains[target]
    changed, _, _, _ = _fixture(base, "xy", invalid=True)
    second = RuleContext(network, changed, num=2)
    assert digit in contract_cluster(second, cluster).supported_domains[target]
    assert not detect_xy_wings(second).clusters
