"""Logical proof detection, independent local support checks, and Kraken cases."""
from collections.abc import Mapping
from dataclasses import replace

import pytest

from demonstrations.sudoku.inferenceClusters.core import RuleContext, RuleLimits, contract_cluster
from demonstrations.sudoku.inferenceClusters.fish_general import detect_general_fish
from demonstrations.sudoku.inferenceClusters.forcing import (
    describe_kraken_fish, describe_kraken_type2, detect_forcing_chains,
    detect_forcing_nets, detect_kraken_fish, detect_kraken_type2,
)
from demonstrations.sudoku.inferenceClusters.implication import (
    build_implication_graph, describe_implication_paths, implication_path,
)
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.sudoku_tests.standard_constraints import get_sudoku_constraint_network


def _context(num, domains, transpose=False, limits=None):
    network = ConstraintNetwork(get_sudoku_constraint_network(num=num))
    names = RuleContext(network, num=num)
    encoded = {}
    for key, values in domains.items():
        if len(key) == 2 and key[0] == "pos":
            row, col = key[1]
            if transpose:
                row, col = col, row
            category = names.pos(row, col)
        else:
            kind, index, digit = key
            if transpose:
                kind = "col" if kind == "row" else "row"
            category = names.unit(kind, index, digit)
        encoded[category] = values
    state = network.create_mac_state(encoded)
    assert state.propagate()
    return network, state, RuleContext(network, state.domains, num=num, limits=limits), encoded


def _exact(context, cluster):
    """Exhaust small original categorical assignments with local pruning."""
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
                supported[output].update(possible[output] if output in possible else (assignment[output],))
            return
        category = order[index]
        for value in context.domains[category]:
            trials[0] += 1
            assert trials[0] <= 2_000_000
            allowed = dict(possible)
            for core, atom in links[category]:
                allowed[atom] = tuple(state for state in allowed[atom]
                                      if core[{category: value, atom: state}] != 0)
                if not allowed[atom]:
                    break
            else:
                assignment[category] = value
                exhaust(index + 1, assignment, allowed)

    exhaust(0, {}, {atom: context.domains[atom] for atom in atoms})
    return {color: tuple(sorted(values)) for color, values in supported.items()}


@pytest.mark.parametrize("detector", (detect_forcing_chains, detect_forcing_nets))
@pytest.mark.parametrize("transpose", [False, True])
def test_exhaustive_source_verities_add_power_beyond_gac(detector, transpose):
    domains = {("pos", (0, 0)): (0, 1), ("pos", (0, 2)): (0, 2), ("pos", (2, 0)): (1, 2)}
    network, state, ctx, encoded = _context(2, domains, transpose)
    source, target = ctx.pos(0, 0), ctx.atom(2, 2, 2)
    before = dict(ctx.domains)
    detection = detector(ctx, sources=(source,))
    cluster = next(cluster for cluster in detection.clusters if cluster.open_colors == (target,))
    result = contract_cluster(ctx, cluster)
    assert result.status == "executed" and state.domains[target] == (0, 1)
    assert result.supported_domains == {target: (0,)} == _exact(ctx, cluster)
    # The added pincer option prevents its forced outgoing digit in one case.
    near_cell = (2, 0) if transpose else (0, 2)
    encoded[ctx.pos(*near_cell)] += (3,)
    sibling = network.create_mac_state(encoded)
    assert sibling.propagate()
    near = RuleContext(network, sibling.domains, num=2)
    assert not any(item.open_colors == (target,) for item in detector(near, sources=(source,)).clusters)
    projected = contract_cluster(near, cluster)
    assert projected.supported_domains == {target: (0, 1)} == _exact(near, cluster)
    assert dict(ctx.domains) == before


@pytest.mark.parametrize("transpose", [False, True])
def test_net_combines_false_premises_unavailable_to_original_conjugate_chains(transpose):
    domains = {("pos", (1, 0)): (0, 1), ("pos", (3, 0)): (0, 1), ("row", 0, 0): (0, 1, 4)}
    network, state, ctx, encoded = _context(3, domains, transpose)
    source_cell, target_cell = ((0, 1), (4, 3)) if transpose else ((1, 0), (3, 4))
    source, target = ctx.pos(*source_cell), ctx.atom(*target_cell, 0)
    chains = detect_forcing_chains(ctx, sources=(source,))
    assert not any(item.open_colors == (target,) for item in chains.clusters)
    nets = detect_forcing_nets(ctx, sources=(source,))
    cluster = next(item for item in nets.clusters if item.open_colors == (target,))
    result = contract_cluster(ctx, cluster)
    assert state.domains[target] == (0, 1)
    assert result.status == "executed" and result.supported_domains == {target: (0,)}
    assert result.supported_domains == _exact(ctx, cluster)
    unit = ctx.unit("col" if transpose else "row", 0, 0)
    encoded[unit] += (8,)
    sibling = network.create_mac_state(encoded)
    assert sibling.propagate()
    near = RuleContext(network, sibling.domains, num=3)
    assert not any(item.open_colors == (target,) for item in detect_forcing_nets(near, sources=(source,)).clusters)
    assert contract_cluster(near, cluster).supported_domains == {target: (0, 1)}


@pytest.mark.parametrize("detector", (detect_forcing_chains, detect_forcing_nets))
def test_refuting_every_source_case_yields_an_exact_contradiction(detector):
    domains = {("pos", (0, col)): (0, 1) for col in (0, 3, 6)}
    network, _, ctx, encoded = _context(3, domains)
    source = ctx.pos(0, 0)
    detection = detector(ctx, sources=(source,))
    cluster = next(item for item in detection.clusters if item.open_colors == (source,))
    result = contract_cluster(ctx, cluster)
    assert result.status == "executed" and result.contradiction
    assert result.supported_domains == {source: ()} == _exact(ctx, cluster)
    encoded[ctx.pos(0, 6)] += (2,)
    sibling = network.create_mac_state(encoded)
    assert sibling.propagate()
    near = RuleContext(network, sibling.domains, num=3)
    assert not any(item.open_colors == (source,) for item in detector(near, sources=(source,)).clusters)
    assert not contract_cluster(near, cluster).contradiction


@pytest.mark.parametrize("limits", (RuleLimits(max_steps=1), RuleLimits(max_chain_length=1),
                                    RuleLimits(max_net_categories=1), RuleLimits(max_graph_nodes=1)))
def test_unfinished_branches_do_not_produce_common_consequences(limits):
    domains = {("pos", (0, 0)): (0, 1), ("pos", (0, 2)): (0, 2), ("pos", (2, 0)): (1, 2)}
    _, _, ctx, _ = _context(2, domains, limits=limits)
    detection = detect_forcing_nets(ctx, sources=(ctx.pos(0, 0),))
    assert detection.truncated and not detection.clusters and detection.steps <= limits.max_steps


def _kraken(kind, transpose):
    domains = {("row", 0, 0): (0, 4, 8), ("row", 3, 0): (0, 4)}
    if kind == 1:
        domains["row", 1, 0] = (4, 8)
        target_cell, target_digit = (4, 4), 0
        near_key, extra = ("row", 1, 0), 1
    else:
        domains["pos", (4, 4)] = (0, 1)
        domains["pos", (1, 7)] = (0, 1)
        target_cell, target_digit = (4, 7), 1
        near_key, extra = ("pos", (4, 4)), 2
    network, state, ctx, encoded = _context(3, domains, transpose)
    kind_base, kind_cover = ("col", "row") if transpose else ("row", "col")
    target_cell = target_cell[::-1] if transpose else target_cell
    target = ctx.atom(*target_cell, target_digit)
    bases, covers = ((kind_base, 0), (kind_base, 3)), ((kind_cover, 0), (kind_cover, 4))
    if near_key[0] == "pos":
        near_category = ctx.pos(*(near_key[1][::-1] if transpose else near_key[1]))
    else:
        near_category = ctx.unit(kind_base, near_key[1], near_key[2])
    return network, state, ctx, encoded, target, bases, covers, near_category, extra


@pytest.mark.parametrize("kind", [1, 2])
@pytest.mark.parametrize("transpose", [False, True])
def test_kraken_chains_compound_with_fish_original_support(kind, transpose):
    network, state, ctx, domains, target, bases, covers, near_category, extra = _kraken(kind, transpose)
    detector = detect_kraken_fish if kind == 1 else detect_kraken_type2
    found = detector(ctx, sizes=(2,), digit=0, bases=bases, covers=covers, targets=(target,))
    assert found.clusters and not found.truncated
    cluster = found.clusters[0]
    result = contract_cluster(ctx, cluster)
    assert result.status == "executed" and not result.contradiction
    assert state.domains[target] == (0, 1) and result.supported_domains == {target: (0,)}
    assert result.supported_domains == _exact(ctx, cluster)
    if kind == 1:
        # The target does not see the fin: ordinary fin visibility finds none.
        ordinary = detect_general_fish(ctx, sizes=(2,), digit=0, bases=bases, covers=covers)
        target_pos = ctx.pos(4, 4)
        assert not any(item.open_colors == (target_pos,) for item in ordinary.clusters)
    else:
        assert dict(cluster.evidence)["cover"] == ("row" if transpose else "col", 4)
        assert target.endswith("_1")  # A different digit from the fish digit.
    domains[near_category] += (extra,)
    sibling = network.create_mac_state(domains)
    assert sibling.propagate()
    near = RuleContext(network, sibling.domains, num=3)
    invalid = detector(near, sizes=(2,), digit=0, bases=bases, covers=covers, targets=(target,))
    assert not invalid.clusters
    projected = contract_cluster(near, cluster)
    assert projected.status == "executed" and projected.supported_domains == {target: (0, 1)}
    assert projected.supported_domains == _exact(near, cluster)


def test_kraken_descriptions_require_every_current_fin_and_case():
    _, _, ctx, _, target, bases, covers, _, _ = _kraken(1, False)
    categories = tuple(ctx.unit(*house, 0) for house in (*bases, *covers))
    atoms = tuple(sorted({atom for category in categories for atom in ctx.categories[category]}))
    fish = ctx.describe("fish", categories, atoms=atoms, outputs=(),
                        evidence=(("digit", 0), ("bases", bases), ("covers", covers),
                                  ("exo_fins", ((0, 8),)), ("endo_fins", ())))
    with pytest.raises(ValueError, match="exhaust"):
        describe_kraken_fish(ctx, fish, {}, outputs=(target,))
    graph = build_implication_graph(ctx)
    fin = ctx.atom(0, 8, 0)
    path = implication_path(graph, fin, True, target, False)
    proof = describe_implication_paths(ctx, (path,), outputs=(target,))
    description = describe_kraken_fish(ctx, fish, {fin: proof}, outputs=(target,))
    assert contract_cluster(ctx, description).supported_domains == {target: (0,)}
    with pytest.raises(ValueError, match="exhaust"):
        describe_kraken_type2(ctx, fish, ("col", 4), {fin: proof}, outputs=(target,))
    with pytest.raises(ValueError):
        describe_kraken_type2(ctx, fish, iter(("col", 4)), {}, outputs=(target,))
    changed = {ctx.unit("row", 0, 0): (0, 4, 7, 8),
               ctx.unit("row", 3, 0): (0, 4), ctx.unit("row", 1, 0): (4, 8)}
    with pytest.raises(ValueError, match="snapshot"):
        describe_kraken_fish(RuleContext(ctx.cores, changed, num=3), fish, {fin: proof}, outputs=(target,))


def test_kraken_and_forcing_options_and_scan_caps_are_explicit():
    _, _, ctx, _, _, _, _, _, _ = _kraken(1, False)
    low = RuleContext(ctx.cores, ctx.domains, num=3, limits=RuleLimits(max_steps=1))
    for detector in (detect_kraken_fish, detect_kraken_type2):
        found = detector(low)
        assert found.truncated and found.steps <= 1 and not found.clusters
    for detector in (detect_forcing_chains, detect_forcing_nets):
        with pytest.raises(TypeError):
            detector(ctx, sources=iter((ctx.pos(0, 0),)))
        with pytest.raises(ValueError):
            detector(ctx, sources=("unknown",))
    with pytest.raises(TypeError):
        detect_kraken_fish(ctx, targets=iter((ctx.atom(4, 4, 0),)))
    with pytest.raises(ValueError):
        detect_kraken_fish(ctx, kind=3)


def test_forcing_catalog_stops_before_inserting_a_node_above_its_cap(monkeypatch):
    ctx = RuleContext(get_sudoku_constraint_network(num=2), num=2,
                      limits=RuleLimits(max_graph_nodes=1))
    original, calls = ctx.options, []

    def observed(category):
        calls.append(category)
        return original(category)

    monkeypatch.setattr(ctx, "options", observed)
    for detector in (detect_forcing_chains, detect_forcing_nets):
        calls.clear()
        detection = detector(ctx)
        assert detection.truncated and not detection.clusters
        assert len(calls) == 1


def test_kraken_reconstructs_required_body_and_bounds_metadata_before_iteration():
    _, _, ctx, _, target, bases, covers, _, _ = _kraken(1, False)
    fin = ctx.atom(0, 8, 0)
    metadata = (("digit", 0), ("bases", bases), ("covers", covers),
                ("exo_fins", ((0, 8),)), ("endo_fins", ()))
    incomplete = ctx.describe("incomplete_fish", (ctx.unit("row", 0, 0),),
                              atoms=(fin,), outputs=(), evidence=metadata)
    graph = build_implication_graph(ctx)
    path = implication_path(graph, fin, True, target, False)
    proof = describe_implication_paths(ctx, (path,), outputs=(target,))
    repaired = describe_kraken_fish(ctx, incomplete, {fin: proof}, outputs=(target,))
    categories = tuple(ctx.unit(*house, 0) for house in (*bases, *covers))
    atoms = tuple(sorted({atom for category in categories for atom in ctx.categories[category]}))
    body = ctx.describe("required", categories, atoms=atoms, outputs=())
    assert set(body.core_keys) <= set(repaired.core_keys)
    assert contract_cluster(ctx, repaired).supported_domains == {target: (0,)}
    bad = dict(metadata)
    bad["exo_fins"] = iter(((0, 8),))
    with pytest.raises(ValueError, match="metadata"):
        describe_kraken_fish(ctx, replace(incomplete, evidence=tuple(bad.items())), {fin: proof}, outputs=(target,))
    bad = dict(metadata)
    bad["covers"] = (("row", 0), ("col", 4))
    with pytest.raises(ValueError, match="distinct"):
        describe_kraken_fish(ctx, replace(incomplete, evidence=tuple(bad.items())), {fin: proof}, outputs=(target,))

    class OversizedMapping(Mapping):
        def __len__(self):
            return ctx.limits.max_patterns

        def __iter__(self):
            pytest.fail("Oversized proof mapping was iterated before its length check")

        def __getitem__(self, key):
            pytest.fail("Oversized proof mapping was read before its length check")

    with pytest.raises(ValueError, match="max_patterns"):
        describe_kraken_fish(ctx, incomplete, OversizedMapping(), outputs=(target,))
