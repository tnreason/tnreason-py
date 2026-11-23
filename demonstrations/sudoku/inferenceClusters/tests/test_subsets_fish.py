"""Independent domain fixtures for the standalone classical rule selectors."""
from itertools import product

import pytest

from demonstrations.sudoku.inferenceClusters.core import RuleContext, RuleLimits, contract_cluster
from demonstrations.sudoku.inferenceClusters.subsets_fish import (
    detect_fish, detect_locked, detect_singles, detect_subsets,
)
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.sudoku_tests.standard_constraints import get_sudoku_constraint_network


@pytest.fixture(scope="module")
def network():
    return ConstraintNetwork(get_sudoku_constraint_network(num=2))


def _closed(network, domains, num=2):
    state = network.create_mac_state(domains)
    assert state.propagate()
    return state, RuleContext(network, state.domains, num=num)


def _find(detection, categories):
    matches = [cluster for cluster in detection.clusters
               if {c for c in cluster.participating_variables if not c.startswith("a_")} == set(categories)]
    assert matches
    return matches[0]


def _exact(context, cluster, result):
    """Enumerate category assignments and each local Boolean support directly.

    This oracle does not use rule preconditions, options(), or the contraction
    implementation.  Boolean atoms are independent after fixing categories.
    """
    categories = tuple(c for c in cluster.participating_variables if not c.startswith("a_"))
    atoms = tuple(c for c in cluster.participating_variables if c.startswith("a_"))
    cores = [context.cores[key] for key in cluster.core_keys]
    supports = {color: set() for color in cluster.open_colors}
    for values in product(*(context.domains[c] for c in categories)):
        assignment = dict(zip(categories, values))
        possible = {}
        for atom in atoms:
            links = [core for core in cores if atom in core.colors]
            allowed = []
            for value in context.domains[atom]:
                assignment[atom] = value
                if all(core[{color: assignment[color] for color in core.colors}] != 0 for core in links):
                    allowed.append(value)
            possible[atom] = allowed
        if not all(possible.values()):
            continue
        for color in supports:
            supports[color].update(possible[color] if color in possible else (assignment[color],))
    assert result.status == "executed"
    assert result.supported_domains == {color: tuple(sorted(values)) for color, values in supports.items()}


@pytest.mark.parametrize("hidden", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
def test_singles_exactly_match_original_gac(network, hidden, transpose):
    naming = RuleContext(network, num=2)
    category = naming.unit("col" if transpose else "row", 0, 1) if hidden else naming.pos(0, 0)
    context = RuleContext(network, {category: (1,)}, num=2)
    cluster = _find(detect_singles(context), (category,))
    result = contract_cluster(context, cluster)
    _exact(context, cluster, result)
    state = network.create_mac_state({category: (1,)})
    assert state.propagate()
    assert all(state.domains[color] == allowed for color, allowed in result.supported_domains.items())
    assert set(result.supported_domains.values()) == {(0,), (1,)}
    near = RuleContext(network, {category: (1, 2)}, num=2)
    assert not any(category in item.participating_variables for item in detect_singles(near).clusters)


@pytest.mark.parametrize("line_kind", ["row", "col"])
@pytest.mark.parametrize("pointing", [False, True])
def test_locked_exact_support_adds_power_beyond_gac(network, line_kind, pointing):
    naming = RuleContext(network, num=2)
    line, box = naming.unit(line_kind, 0, 0), naming.unit("square", 0, 0)
    box_allowed = (0, 1) if line_kind == "row" else (0, 2)
    source, target, allowed, expected = ((box, line, box_allowed, (0, 1)) if pointing
                                         else (line, box, (0, 1), box_allowed))
    state, context = _closed(network, {source: allowed})
    assert state.domains[target] == tuple(range(4))
    cluster = _find(detect_locked(context), (line, box))
    result = contract_cluster(context, cluster)
    assert result.supported_domains[target] == expected
    _exact(context, cluster, result)
    outside = next(value for value in range(4) if value not in allowed)
    _, near = _closed(network, {source: (*allowed, outside)})
    assert not any(set(item.open_colors) == {target}
                   and {c for c in item.participating_variables if not c.startswith("a_")} == {line, box}
                   for item in detect_locked(near).clusters)


@pytest.mark.parametrize("hidden", [False, True])
@pytest.mark.parametrize("line_kind", ["row", "col"])
def test_pairs_exact_support_adds_power_beyond_gac(network, hidden, line_kind):
    naming = RuleContext(network, num=2)
    cells = ((0, 0), (0, 2)) if line_kind == "row" else ((0, 0), (2, 0))
    positions = tuple(naming.pos(*cell) for cell in cells)
    units = tuple(naming.unit(line_kind, 0, digit) for digit in (0, 1))
    source, allowed, targets, expected = ((units, (0, 2), positions, (0, 1)) if hidden
                                          else (positions, (0, 1), units, (0, 2)))
    state, context = _closed(network, {category: allowed for category in source})
    assert all(state.domains[target] == tuple(range(4)) for target in targets)
    cluster = _find(detect_subsets(context, hidden=hidden, sizes=(2,)), positions + units)
    result = contract_cluster(context, cluster)
    assert all(result.supported_domains[target] == expected for target in targets)
    _exact(context, cluster, result)
    _, near = _closed(network, {source[0]: allowed, source[1]: (*allowed, 3)})
    assert not any({c for c in item.participating_variables if not c.startswith("a_")}
                   == set(positions + units) for item in detect_subsets(near, hidden=hidden, sizes=(2,)).clusters)


@pytest.mark.parametrize("hidden", [False, True])
@pytest.mark.parametrize("line_kind", ["row", "col"])
@pytest.mark.parametrize("size", [3, 4])
def test_generalized_subsets_have_unequal_candidate_sets(hidden, line_kind, size):
    network = ConstraintNetwork(get_sudoku_constraint_network(num=3))
    naming = RuleContext(network, num=3)
    offsets = (0, 3, 6, 7)[:size]
    cells = tuple((0, offset) if line_kind == "row" else (offset, 0) for offset in offsets)
    positions = tuple(naming.pos(*cell) for cell in cells)
    units = tuple(naming.unit(line_kind, 0, digit) for digit in range(size))
    if hidden:
        domains = {unit: (offsets[i], offsets[(i + 1) % size]) for i, unit in enumerate(units)}
    else:
        domains = {position: (i, (i + 1) % size) for i, position in enumerate(positions)}
    state, context = _closed(network, domains, num=3)
    cluster = _find(detect_subsets(context, hidden=hidden, sizes=(size,)), positions + units)
    result = contract_cluster(context, cluster)
    targets = positions if hidden else units
    expected = ({position: tuple(digit for digit, unit in enumerate(units)
                                 if offsets[i] in domains[unit])
                 for i, position in enumerate(positions)} if hidden else
                {unit: tuple(offsets[i] for i, position in enumerate(positions)
                             if digit in domains[position])
                 for digit, unit in enumerate(units)})
    assert result.status == "executed"
    assert all(result.supported_domains[target] == expected[target] for target in targets)
    assert any(set(state.domains[target]) - set(expected[target]) for target in targets)
    # Break the cardinality equality without introducing a single.
    domains[next(iter(domains))] += ((8 if hidden else size),)
    _, near = _closed(network, domains, num=3)
    assert not any({c for c in item.participating_variables if not c.startswith("a_")}
                   == set(positions + units)
                   for item in detect_subsets(near, hidden=hidden, sizes=(size,)).clusters)


@pytest.mark.parametrize("kind", ["basic", "finned", "sashimi"])
@pytest.mark.parametrize("transpose", [False, True])
def test_fish_exact_support_adds_power_beyond_gac(network, kind, transpose):
    naming = RuleContext(network, num=2)
    base_kind, cover_kind = ("col", "row") if transpose else ("row", "col")
    bases = tuple(naming.unit(base_kind, index, 0) for index in (0, 2))
    covers = tuple(naming.unit(cover_kind, index, 0) for index in (0, 2))
    allowed = {"basic": (0, 2), "finned": (0, 1, 2), "sashimi": (1, 2)}[kind]
    state, context = _closed(network, {bases[0]: allowed, bases[1]: (0, 2)})
    categories = bases + covers + (() if kind == "basic" else (naming.unit("square", 0, 0),))
    cluster = _find(detect_fish(context, sizes=(2,), kind=kind), categories)
    result = contract_cluster(context, cluster)
    assert 1 in state.domains[covers[0]] and 1 not in result.supported_domains[covers[0]]
    _exact(context, cluster, result)
    # The added candidate is in a different box from the fin.  A cover target
    # can now be true, and the original pattern must not be recognized.
    _, near = _closed(network, {bases[0]: (*allowed, 3), bases[1]: (0, 2)})
    for item in detect_fish(near, sizes=(2,), kind=kind).clusters:
        if covers[0] in item.open_colors:
            projected = contract_cluster(near, item)
            assert projected.status == "executed"
            assert 1 in projected.supported_domains[covers[0]]


@pytest.mark.parametrize("size", [3, 4])
@pytest.mark.parametrize("transpose", [False, True])
def test_swordfish_and_jellyfish(size, transpose):
    network = ConstraintNetwork(get_sudoku_constraint_network(num=3))
    naming = RuleContext(network, num=3)
    base_kind, cover_kind = ("col", "row") if transpose else ("row", "col")
    indices = (0, 3, 6, 7)[:size]
    bases = tuple(naming.unit(base_kind, index, 0) for index in indices)
    covers = tuple(naming.unit(cover_kind, index, 0) for index in indices)
    domains = {base: (indices[i], indices[(i + 1) % size]) for i, base in enumerate(bases)}
    state, context = _closed(network, domains, num=3)
    cluster = _find(detect_fish(context, sizes=(size,)), bases + covers)
    result = contract_cluster(context, cluster)
    assert result.status == "executed"
    expected = {cover: tuple(index for index, base in zip(indices, bases)
                             if cover_index in domains[base])
                for cover_index, cover in zip(indices, covers)}
    assert all(result.supported_domains[cover] == expected[cover] for cover in covers)
    assert any(set(state.domains[cover]) - set(indices) for cover in covers)
    domains[bases[0]] += (8,)
    _, near = _closed(network, domains, num=3)
    assert not any({c for c in item.participating_variables if not c.startswith("a_")}
                   == set(bases + covers) for item in detect_fish(near, sizes=(size,)).clusters)


@pytest.mark.parametrize("detector", [detect_singles, detect_locked, detect_subsets, detect_fish])
def test_detection_work_is_bounded_and_inputs_are_immutable(network, detector):
    context = RuleContext(network, num=2, limits=RuleLimits(max_steps=1))
    before = dict(context.domains)
    result = detector(context)
    assert result.steps <= 1
    # Empty fish/subset searches can finish without trying a combination.
    assert result.truncated or not result.clusters
    assert dict(context.domains) == before


def test_pattern_cap_and_invalid_requests(network):
    naming = RuleContext(network, num=2)
    context = RuleContext(network, {naming.pos(0, 0): (0,), naming.pos(2, 2): (1,)},
                          num=2, limits=RuleLimits(max_patterns=1))
    detection = detect_singles(context)
    assert len(detection.clusters) == 1 and detection.truncated
    for detector in (detect_subsets, detect_fish):
        with pytest.raises(ValueError):
            detector(context, sizes=(5,))
        with pytest.raises(ValueError):
            detector(context, sizes=(True,))
        with pytest.raises(TypeError):
            detector(context, sizes=iter((2,)))
    with pytest.raises(ValueError):
        detect_fish(context, kind="mutant")
