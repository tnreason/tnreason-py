"""Explicit independent mixed-house and advanced-fin fixtures."""
import pytest

from demonstrations.sudoku.inferenceClusters.core import RuleContext, RuleLimits, contract_cluster
from demonstrations.sudoku.inferenceClusters.fish_general import (
    detect_cannibalistic_fish, detect_endo_fish, detect_franken_fish,
    detect_general_fish, detect_multi_fin_fish, detect_mutant_fish, detect_siamese_fish,
)
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.sudoku_tests.standard_constraints import get_sudoku_constraint_network


_CASES = {
    "franken": (3, (("row", 0), ("square", 6)), (("col", 0), ("col", 4)),
                {("row", 0): (0, 4), ("square", 6): (0, 3, 6)}, (4, 4), (("row", 0), 8)),
    "mutant": (3, (("row", 3), ("col", 1)), (("col", 5), ("square", 3)),
               {("row", 3): (0, 5), ("col", 1): (4, 5)}, (4, 5), (("row", 3), 8)),
    "endo": (2, (("row", 0), ("square", 0)), (("col", 0), ("col", 2)),
             {("row", 0): (0, 2), ("square", 0): (0, 2)}, (2, 0), (("square", 0), 3)),
    "cannibalistic": (3, (("row", 0), ("row", 1), ("row", 3)),
                      (("col", 0), ("col", 4), ("square", 0)),
                      {("row", 0): (0, 1, 4), ("row", 1): (0, 1, 4), ("row", 3): (0, 4)},
                      (0, 0), (("row", 3), 8)),
    "multi_fin": (3, (("col", 8), ("row", 8), ("col", 3)),
                  (("row", 2), ("square", 4), ("square", 8)),
                  {("col", 8): (2, 4), ("row", 8): (4, 6, 7), ("col", 3): (2, 3, 5)},
                  (4, 4), (("col", 8), 0)),
}
_DETECTORS = {"franken": detect_franken_fish, "mutant": detect_mutant_fish,
              "endo": detect_endo_fish, "cannibalistic": detect_cannibalistic_fish,
              "multi_fin": detect_multi_fin_fish}


def _transpose_house(house, num):
    kind, index = house
    if kind == "square":
        return kind, index % num * num + index // num
    return ("col" if kind == "row" else "row"), index


def _transpose_value(house, value, num):
    return value % num * num + value // num if house[0] == "square" else value


def _fixture(name, transpose):
    num, bases, covers, domains, target, near = _CASES[name]
    if transpose:
        bases = tuple(_transpose_house(house, num) for house in bases)
        covers = tuple(_transpose_house(house, num) for house in covers)
        domains = {_transpose_house(house, num): tuple(_transpose_value(house, value, num) for value in values)
                   for house, values in domains.items()}
        target = target[::-1]
        near = _transpose_house(near[0], num), _transpose_value(near[0], near[1], num)
    network = ConstraintNetwork(get_sudoku_constraint_network(num=num))
    names = RuleContext(network, num=num)
    encoded = {names.unit(*house, 0): values for house, values in domains.items()}
    state = network.create_mac_state(encoded)
    assert state.propagate()
    context = RuleContext(network, state.domains, num=num)
    return network, state, context, bases, covers, target, near, encoded


def _exact_original_projection(context, cluster):
    """Exhaust categorical assignments, pruning only realized original links.

    After a category is fixed, each incident Boolean atom retains the states
    supported by every assigned original factor. This is exhaustive local
    support verification; no Sudoku rule predicate or completed grid enters it.
    """
    categories = tuple(c for c in cluster.participating_variables if not c.startswith("a_"))
    atoms = tuple(c for c in cluster.participating_variables if c.startswith("a_"))
    links = {category: [] for category in categories}
    for key in cluster.core_keys:
        core = context.cores[key]
        category = next(color for color in core.colors if color in links)
        atom = next(color for color in core.colors if color.startswith("a_"))
        links[category].append((core, atom))
    order = sorted(categories, key=lambda category: (len(context.domains[category]), -len(links[category])))
    supported = {color: set() for color in cluster.open_colors}
    trials = [0]

    def exhaust(index, assignment, possible):
        if index == len(order):
            for output in supported:
                supported[output].add(assignment[output])
            return
        category = order[index]
        for value in context.domains[category]:
            trials[0] += 1
            assert trials[0] <= 1_000_000, "Independent oracle exceeded its fixture bound"
            allowed = dict(possible)
            valid = True
            for core, atom in links[category]:
                retained = tuple(state for state in allowed[atom]
                                 if core[{category: value, atom: state}] != 0)
                if not retained:
                    valid = False
                    break
                allowed[atom] = retained
            if valid:
                assignment[category] = value
                exhaust(index + 1, assignment, allowed)

    exhaust(0, {}, {atom: context.domains[atom] for atom in atoms})
    return {color: tuple(sorted(values)) for color, values in supported.items()}


@pytest.mark.parametrize("name", tuple(_CASES))
@pytest.mark.parametrize("transpose", [False, True])
def test_mixed_fish_eliminations_exact_support_and_invalid_near(name, transpose):
    network, state, ctx, bases, covers, target, near, domains = _fixture(name, transpose)
    original_domains = dict(ctx.domains)
    original_values = {key: core.values.copy() for key, core in network.coresDict.items()}
    detection = _DETECTORS[name](ctx, sizes=(len(bases),), digit=0, bases=bases, covers=covers)
    target_name = ctx.pos(*target)
    matches = [cluster for cluster in detection.clusters if cluster.open_colors == (target_name,)]
    assert matches and not detection.truncated
    cluster = matches[0]
    result = contract_cluster(ctx, cluster)
    assert result.status == "executed" and not result.contradiction
    assert 0 in state.domains[target_name] and 0 not in result.supported_domains[target_name]
    assert result.supported_domains == _exact_original_projection(ctx, cluster)
    evidence = dict(cluster.evidence)
    if name in ("franken", "mutant"):
        assert evidence["taxonomy"] == name
    if name == "endo":
        assert evidence["endo_fins"]
    if name == "cannibalistic":
        assert evidence["cannibalistic"]
    if name == "multi_fin":
        assert len({(row // ctx.num, col // ctx.num) for row, col in evidence["exo_fins"]}) > 1
    domains[ctx.unit(*near[0], 0)] += (near[1],)
    near_state = network.create_mac_state(domains)
    assert near_state.propagate()
    near_ctx = RuleContext(network, near_state.domains, num=ctx.num)
    invalid = _DETECTORS[name](near_ctx, sizes=(len(bases),), digit=0, bases=bases, covers=covers)
    assert not any(item.open_colors == (target_name,) for item in invalid.clusters)
    # Even explicitly contracting the old structural cluster cannot leak its
    # domain-conditioned elimination into this larger-domain sibling.
    near_result = contract_cluster(near_ctx, cluster)
    assert near_result.status == "executed" and 0 in near_result.supported_domains[target_name]
    assert near_result.supported_domains == _exact_original_projection(near_ctx, cluster)
    assert dict(ctx.domains) == original_domains
    assert all((network.coresDict[key].values == value).all() for key, value in original_values.items())


@pytest.mark.parametrize("transpose", [False, True])
def test_siamese_composite_projects_both_distinct_component_targets(transpose):
    network = ConstraintNetwork(get_sudoku_constraint_network(num=2))
    ctx = RuleContext(network, num=2)
    kind, cover_kind = ("col", "row") if transpose else ("row", "col")
    bases = ((kind, 0), (kind, 2))
    variants = (((cover_kind, 0), (cover_kind, 2)), ((cover_kind, 0), (cover_kind, 3)))
    domains = {ctx.unit(kind, 0, 0): (0, 2), ctx.unit(kind, 2, 0): (0, 3)}
    state = network.create_mac_state(domains)
    assert state.propagate()
    ctx = RuleContext(network, state.domains, num=2)
    target_cells = ((3, 2), (1, 3))
    if transpose:
        target_cells = tuple(cell[::-1] for cell in target_cells)
    target_names = {ctx.pos(*cell) for cell in target_cells}
    detection = detect_siamese_fish(ctx, sizes=(2,), digit=0, bases=bases, cover_pairs=variants)
    cluster = next(cluster for cluster in detection.clusters if set(cluster.open_colors) == target_names)
    result = contract_cluster(ctx, cluster)
    assert result.status == "executed"
    assert all(0 in state.domains[target] and 0 not in result.supported_domains[target] for target in target_names)
    assert result.supported_domains == _exact_original_projection(ctx, cluster)
    domains[ctx.unit(kind, 0, 0)] += (1,)
    near_state = network.create_mac_state(domains)
    assert near_state.propagate()
    near_ctx = RuleContext(network, near_state.domains, num=2)
    assert not detect_siamese_fish(near_ctx, sizes=(2,), digit=0, bases=bases, cover_pairs=variants).clusters
    projected = contract_cluster(near_ctx, cluster)
    assert projected.status == "executed"
    assert all(0 in projected.supported_domains[target] for target in target_names)
    assert projected.supported_domains == _exact_original_projection(near_ctx, cluster)


def test_fins_and_overlap_are_explicit_options():
    _, _, ctx, bases, covers, _, _, _ = _fixture("endo", False)
    assert not detect_endo_fish(ctx, sizes=(2,), digit=0, bases=bases, covers=covers, endo=False).clusters
    assert not detect_endo_fish(ctx, sizes=(2,), digit=0, bases=bases, covers=covers, fins=False).clusters
    assert not detect_endo_fish(ctx, sizes=(2,), digit=0, bases=bases, covers=covers, max_fins=0).clusters
    _, _, ctx, bases, covers, _, _, _ = _fixture("multi_fin", False)
    assert not detect_multi_fin_fish(ctx, sizes=(3,), digit=0, bases=bases, covers=covers, max_fins=1).clusters


@pytest.mark.parametrize("detector", (detect_general_fish, detect_franken_fish, detect_mutant_fish,
                                      detect_multi_fin_fish, detect_endo_fish, detect_cannibalistic_fish,
                                      detect_siamese_fish))
def test_automatic_house_combinations_respect_scan_bound(detector):
    ctx = RuleContext(get_sudoku_constraint_network(num=2), num=2, limits=RuleLimits(max_steps=5))
    detection = detector(ctx)
    assert detection.steps == 5 and detection.truncated and not detection.clusters


def test_pattern_execution_and_composite_bounds_are_separate():
    _, _, original, bases, covers, _, _, _ = _fixture("mutant", False)
    ctx = RuleContext(original.cores, original.domains, num=3,
                      limits=RuleLimits(max_patterns=1, max_work=1))
    detection = detect_mutant_fish(ctx, sizes=(2,), digit=0, bases=bases, covers=covers)
    assert len(detection.clusters) == 1 and detection.truncated
    assert contract_cluster(ctx, detection.clusters[0]).status == "skipped_budget"
    with pytest.raises(ValueError):
        detect_siamese_fish(ctx, max_components=ctx.limits.max_steps + 1)
    for options in ({"sizes": iter((2,))}, {"bases": iter(bases)}, {"digit": True},
                    {"max_fins": ctx.n ** 2 + 1}, {"sizes": (True,)},
                    {"covers": (("row", 0), ("row", 0))}):
        with pytest.raises((TypeError, ValueError)):
            detect_general_fish(ctx, **options)
