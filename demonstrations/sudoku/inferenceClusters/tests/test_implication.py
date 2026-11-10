"""Finite independent fixtures for implication-selected original tensor motifs."""
from dataclasses import replace
from itertools import product, repeat

import pytest

from demonstrations.sudoku.inferenceClusters.core import RuleContext, RuleLimits, contract_cluster
from demonstrations.sudoku.inferenceClusters.implication import (
    ImplicationEdge, build_implication_graph, describe_implication_paths,
    detect_aic, detect_empty_rectangles, detect_grouped_aic, detect_medusa,
    detect_multi_coloring, detect_nice_loops, detect_skyscrapers,
    detect_turbot_fish, detect_two_string_kites, detect_w_wings, detect_x_chains,
    implication_path,
)
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.sudoku_tests.standard_constraints import get_sudoku_constraint_network


CALLS = {
    "aic": detect_aic, "x_chain": detect_x_chains,
    "turbot": detect_turbot_fish, "kite": detect_two_string_kites,
    "skyscraper": detect_skyscrapers, "w_wing": detect_w_wings,
    "empty_rectangle": detect_empty_rectangles, "grouped": detect_grouped_aic,
    "nice_loop": detect_nice_loops, "multi_coloring": detect_multi_coloring,
    "medusa": detect_medusa,
}


@pytest.fixture(scope="module")
def networks():
    return {num: ConstraintNetwork(get_sudoku_constraint_network(num=num)) for num in (2, 3)}


def _case(name):
    num, positions, units = 2, {}, []
    witness = None
    if name == "aic":
        positions = {(0, 0): (0, 1), (0, 2): (0, 2), (2, 0): (1, 2)}
        bad_positions = {**positions, (0, 2): (0, 2, 3)}
        target = 2, 2, 2
        witness = ((0, 1, 3, 2), (2, 3, 1, 0), (1, 0, 2, 3), (3, 2, 0, 1))
    elif name in ("kite", "turbot", "x_chain", "multi_coloring"):
        units = [("row", 0, 2, (1, 3)), ("col", 0, 2, (1, 2))]
        bad_units = [("row", 0, 2, (1, 2, 3)), units[1]]
        target = 2, 3, 2
    elif name == "skyscraper":
        num = 3
        units = [("row", 0, 0, (0, 3)), ("row", 1, 0, (0, 4))]
        bad_units = [units[0], ("row", 1, 0, (0, 4, 6))]
        target = 2, 3, 0
    elif name == "w_wing":
        positions = {(0, 0): (0, 1), (2, 1): (0, 1)}
        units = [("col", 2, 0, (0, 2))]
        bad_units = [("col", 2, 0, (0, 1, 2))]
        target = 0, 1, 1
    elif name == "empty_rectangle":
        units = [("square", 0, 1, (1, 2)), ("col", 2, 1, (0, 2))]
        bad_units = [units[0], ("col", 2, 1, (0, 2, 3))]
        target = 2, 0, 1
    elif name == "grouped":
        num = 3
        units = [("square", 0, 4, (1, 2, 3, 6)), ("col", 4, 4, (0, 4))]
        bad_units = [("square", 0, 4, (1, 2, 3, 4, 6)), units[1]]
        target = 4, 0, 4
    elif name == "nice_loop":
        units = [("row", 0, 2, (0, 2)), ("row", 2, 2, (0, 2))]
        bad_units = [units[0], ("row", 2, 2, (0, 2, 3))]
        target = 1, 0, 2
    elif name == "medusa":
        positions = {(0, 0): (1, 2), (0, 2): (0, 1, 2)}
        units = [("row", 0, 1, (0, 2)), ("row", 0, 2, (0, 2))]
        bad_units = [units[0], ("row", 0, 2, (0, 1, 2))]
        target = 0, 2, 0
        witness = ((1, 2, 0, 3), (0, 3, 1, 2), (2, 1, 3, 0), (3, 0, 2, 1))
    elif name == "medusa_empty":
        positions = {(0, 1): (0, 1), (1, 0): (0, 1)}
        bad_positions = {**positions, (1, 0): (0, 1, 2)}
        units = [("row", 0, 0, (0, 1))]
        target = 0, 0, 0
    elif name in ("medusa_conflict", "multi_type2"):
        if name == "medusa_conflict":
            positions = {(0, 0): (0, 1), (0, 2): (1, 2), (2, 2): (0, 2)}
            units = [("row", 0, 1, (0, 2)), ("col", 2, 2, (0, 2)),
                     ("row", 2, 0, (0, 2))]
            bad_units = [units[0], ("col", 2, 2, (0, 1, 2)), units[2]]
        else:
            units = [("row", 0, 0, (0, 2)), ("col", 2, 0, (0, 2)),
                     ("row", 3, 0, (0, 3))]
            bad_units = [units[0], units[1], ("row", 3, 0, (0, 1, 3))]
        target = 0, 0, 0
        witness = ((0, 2, 1, 3), (1, 3, 2, 0), (3, 1, 0, 2), (2, 0, 3, 1))
    elif name == "placement":
        units = [("row", 0, 1, (0, 1)), ("col", 0, 1, (0, 1))]
        bad_units = [units[0], ("col", 0, 1, (0, 1, 2))]
        target = 0, 0, 1
    else:
        raise AssertionError(name)
    if witness is None:
        if num == 2:
            witness = ((0, 1, 2, 3), (2, 3, 0, 1), (1, 0, 3, 2), (3, 2, 1, 0))
        else:
            n = num ** 2
            witness = tuple(tuple((row * num + row // num + col) % n for col in range(n))
                            for row in range(n))
    return (num, positions, units, locals().get("bad_positions", positions),
            locals().get("bad_units", units), target, witness)


def _contexts(networks, name, transpose=False, limits=None):
    num, positions, units, bad_positions, bad_units, target, witness = _case(name)
    blank = RuleContext(networks[num], num=num)

    def domains(positions, units):
        result = {blank.pos(*(cell[::-1] if transpose else cell)): values
                  for cell, values in positions.items()}
        for kind, index, digit, values in units:
            if transpose:
                if kind == "square":
                    index = index % num * num + index // num
                    values = tuple(value % num * num + value // num for value in values)
                else:
                    kind = {"row": "col", "col": "row"}[kind]
            result[blank.unit(kind, index, digit)] = values
        return result

    row, col, digit = target
    if transpose:
        row, col = col, row
        witness = tuple(zip(*witness))
    good = RuleContext(networks[num], domains(positions, units), num=num, limits=limits)
    bad = RuleContext(networks[num], domains(bad_positions, bad_units), num=num, limits=limits)
    return good, bad, blank.atom(row, col, digit), witness


def _exact_support(ctx, cluster, target):
    """Enumerate categorical values; existentially check each original atom.

    This independent reference reads realized original entries. It neither uses
    the contraction executor nor searches for a completed Sudoku grid.
    """
    categories = sorted(color for color in cluster.participating_variables if not color.startswith("a_"))
    atoms = sorted(color for color in cluster.participating_variables if color.startswith("a_"))
    tables = {}
    for category in categories:
        incident = {}
        for key in cluster.core_keys:
            core = ctx.cores[key]
            if category not in core.colors:
                continue
            atom = next(color for color in core.colors if color != category)
            incident.setdefault(atom, []).append(core)
        tables[category] = {
            value: tuple((atom, sum(1 << truth for truth in ctx.domains[atom]
                                   if all(core[{category: value, atom: truth}] != 0 for core in links)))
                         for atom, links in incident.items())
            for value in ctx.domains[category]
        }
    result = 0
    for values in product(*(ctx.domains[category] for category in categories)):
        masks = {atom: sum(1 << value for value in ctx.domains[atom]) for atom in atoms}
        possible = True
        for category, value in zip(categories, values):
            for atom, support in tables[category][value]:
                masks[atom] &= support
                if not masks[atom]:
                    possible = False
                    break
            if not possible:
                break
        if possible:
            result |= masks[target]
    return tuple(value for value in (0, 1) if result & (1 << value))


def _witness(ctx, grid):
    assignment = {}
    for row in range(ctx.n):
        for col in range(ctx.n):
            assignment[ctx.pos(row, col)] = grid[row][col]
            for digit in range(ctx.n):
                assignment[ctx.atom(row, col, digit)] = int(grid[row][col] == digit)
    for category, atoms in ctx.categories.items():
        if not category.startswith("pos_"):
            assignment[category] = next(index for index, atom in enumerate(atoms) if assignment[atom])
    assert all(value in ctx.domains[color] for color, value in assignment.items())
    assert all(core[{color: assignment[color] for color in core.colors}] != 0
               for core in ctx.cores.values())
    return assignment


@pytest.mark.parametrize("name", CALLS)
@pytest.mark.parametrize("transpose", [False, True])
def test_named_and_shared_implication_motifs_add_power_and_match_exact_support(networks, name, transpose):
    good, _, target, _ = _contexts(networks, name, transpose)
    before = dict(good.domains)
    ordinary = networks[good.num].create_mac_state(good.domains)
    assert ordinary.propagate() and ordinary.domains[target] == (0, 1)
    detection = CALLS[name](good)
    assert not detection.truncated
    cluster = next(cluster for cluster in detection.clusters if target in cluster.open_colors)
    result = contract_cluster(good, cluster)
    assert result.status == "executed" and not result.contradiction
    assert result.supported_domains[target] == (0,)
    assert result.supported_domains[target] == _exact_support(good, cluster, target)
    assert dict(good.domains) == before
    assert set(cluster.core_keys) <= good.cores.keys()


@pytest.mark.parametrize("name", CALLS)
@pytest.mark.parametrize("transpose", [False, True])
def test_invalid_near_patterns_preserve_independent_complete_witness(networks, name, transpose):
    _, bad, target, grid = _contexts(networks, name, transpose)
    assignment = _witness(bad, grid)
    assert assignment[target] == 1
    for cluster in CALLS[name](bad).clusters:
        if target in cluster.open_colors:
            result = contract_cluster(bad, cluster)
            if result.status == "executed":
                assert 1 in result.supported_domains[target]


@pytest.mark.parametrize("name,call,expected", [
    ("medusa_empty", detect_medusa, "empties"),
    ("medusa_conflict", detect_medusa, "same-color conflict"),
    ("multi_type2", detect_multi_coloring, "type 2"),
])
@pytest.mark.parametrize("transpose", [False, True])
def test_coloring_additional_contradiction_mechanisms(networks, name, call, expected, transpose):
    good, bad, target, grid = _contexts(networks, name, transpose)
    ordinary = networks[good.num].create_mac_state(good.domains)
    assert ordinary.propagate() and ordinary.domains[target] == (0, 1)
    cluster = next(cluster for cluster in call(good).clusters
                   if target in cluster.open_colors and expected in cluster.evidence[0])
    result = contract_cluster(good, cluster)
    assert result.status == "executed" and result.supported_domains[target] == (0,)
    assert result.supported_domains[target] == _exact_support(good, cluster, target)
    assert _witness(bad, grid)[target] == 1
    for cluster in call(bad).clusters:
        if target in cluster.open_colors:
            result = contract_cluster(bad, cluster)
            if result.status == "executed":
                assert 1 in result.supported_domains[target]


@pytest.mark.parametrize("transpose", [False, True])
def test_discontinuous_strong_closure_places_instead_of_eliminates(networks, transpose):
    good, bad, target, grid = _contexts(networks, "placement", transpose)
    for call in (detect_aic, detect_nice_loops):
        cluster = next(cluster for cluster in call(good).clusters if target in cluster.open_colors)
        result = contract_cluster(good, cluster)
        assert result.status == "executed" and result.supported_domains[target] == (1,)
        assert result.supported_domains[target] == _exact_support(good, cluster, target)
        # The same original motif is safe under a different snapshot.
        changed = contract_cluster(bad, cluster)
        assert 0 in changed.supported_domains[target]
    assert _witness(bad, grid)[target] == 0


def test_continuous_and_discontinuous_loop_evidence_are_both_real(networks):
    good, _, target, _ = _contexts(networks, "nice_loop")
    detection = detect_nice_loops(good)
    for kind in ("continuous alternating loop", "discontinuous loop"):
        cluster = next(cluster for cluster in detection.clusters
                       if target in cluster.open_colors and kind in cluster.evidence[0])
        assert contract_cluster(good, cluster).supported_domains[target] == (0,)


@pytest.mark.parametrize("transpose", [False, True])
def test_medusa_unit_exhaustion_extension_has_original_proof_and_invalid_witness(networks, transpose):
    blank = RuleContext(networks[3], num=3)
    kind, perpendicular = ("col", "row") if transpose else ("row", "col")
    source = blank.unit(kind, 3, 0)
    domains = {blank.unit(kind, 0, 0): (0, 5),
               blank.unit(perpendicular, 5, 0): (0, 4), source: (0, 4)}
    good = RuleContext(networks[3], domains, num=3)
    target = blank.atom(0, 0, 0)
    ordinary = networks[3].create_mac_state(domains)
    assert ordinary.propagate() and ordinary.domains[target] == (0, 1)
    # The component's same-color 0s at (0,0) and (4,5) eliminate both
    # candidates of row 3, without a same-color conflict or opposite-color
    # trap in that component. The row's original exactly-one tensors matter.
    cluster = next(cluster for cluster in detect_medusa(good).clusters
                   if target in cluster.open_colors and f"empties {source}" in cluster.evidence[0])
    result = contract_cluster(good, cluster)
    assert result.status == "executed" and result.supported_domains[target] == (0,)
    assert result.supported_domains[target] == _exact_support(good, cluster, target)
    bad = RuleContext(networks[3], {**domains, source: (0, 4, 8)}, num=3)
    grid = tuple(tuple((row * 3 + row // 3 + col) % 9 for col in range(9)) for row in range(9))
    if transpose:
        grid = tuple(zip(*grid))
    assert _witness(bad, grid)[target] == 1
    changed = contract_cluster(bad, cluster)
    assert changed.status == "executed" and 1 in changed.supported_domains[target]


def test_named_shape_restrictions_are_not_generic_aliases(networks):
    kite, _, _, _ = _contexts(networks, "kite")
    sky, _, _, _ = _contexts(networks, "skyscraper")
    assert not detect_skyscrapers(kite).clusters
    assert not detect_two_string_kites(sky).clusters
    assert detect_turbot_fish(kite).clusters and detect_turbot_fish(sky).clusters


def test_group_nodes_are_actual_disjunctions_and_optional(networks):
    good, _, target, _ = _contexts(networks, "grouped")
    graph = build_implication_graph(good, grouped=True)
    assert not graph.truncated
    path = implication_path(graph, target, True, target, False)
    assert path and any(len(edge.first) > 1 or len(edge.second) > 1 for edge in path)
    cluster = describe_implication_paths(good, (path,), (target,))
    assert contract_cluster(good, cluster).supported_domains[target] == (0,)
    singleton = build_implication_graph(good)
    assert implication_path(singleton, target, True, target, False) is None


@pytest.mark.parametrize("name", CALLS)
def test_graph_and_named_detection_obey_scan_budget(networks, name):
    ctx = RuleContext(networks[2], num=2, limits=RuleLimits(max_steps=5))
    detection = CALLS[name](ctx)
    assert detection.steps == 5 and detection.truncated and not detection.clusters


@pytest.mark.parametrize("field", ["max_graph_nodes", "max_graph_edges"])
def test_graph_construction_caps_are_explicit(networks, field):
    good, _, _, _ = _contexts(networks, "aic", limits=replace(RuleLimits(), **{field: 1}))
    graph = build_implication_graph(good)
    assert graph.truncated
    assert len(graph.nodes) <= good.limits.max_graph_nodes
    assert len(graph.edges) <= good.limits.max_graph_edges


def test_coloring_component_and_pattern_bounds(networks):
    good, _, _, _ = _contexts(networks, "medusa", limits=RuleLimits(max_coloring_nodes=1))
    assert detect_medusa(good).truncated
    good, _, _, _ = _contexts(networks, "nice_loop", limits=RuleLimits(max_patterns=1))
    detection = detect_nice_loops(good)
    assert detection.truncated and len(detection.clusters) == 1


def test_public_path_and_description_reject_unbounded_inputs(networks):
    good, _, target, _ = _contexts(networks, "aic")
    graph = build_implication_graph(good)
    for value in (0, -1, True, 1.5, good.limits.max_chain_length + 1):
        with pytest.raises(ValueError):
            implication_path(graph, target, True, target, False, max_length=value)
    with pytest.raises(ValueError):
        implication_path(graph, repeat(target), True, target, False)
    path = implication_path(graph, target, True, target, False)
    assert path
    with pytest.raises(ValueError):
        describe_implication_paths(good, repeat(path), (target,))
    with pytest.raises(ValueError):
        describe_implication_paths(good, (repeat(path[0]),), (target,))
    with pytest.raises(ValueError):
        describe_implication_paths(good, (path,), repeat(target))
    fake = ImplicationEdge(frozenset((target,)), frozenset(("absent",)), "weak", ("absent",))
    with pytest.raises(ValueError):
        describe_implication_paths(good, ((fake,),), (target,))


def test_recognized_pattern_can_skip_execution_without_becoming_a_deduction(networks):
    good, _, _, _ = _contexts(networks, "kite", limits=RuleLimits(max_size=1))
    selection = detect_two_string_kites(good)
    assert selection.clusters
    result = contract_cluster(good, selection.clusters[0])
    assert result.status == "skipped_budget" and not result.summaries
