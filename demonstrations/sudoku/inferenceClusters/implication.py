"""Bounded original-tensor motifs selected with implication graphs.

Graph searches recognize proofs, never apply their logical conclusions. Exact
Boolean contraction of every selected original incidence tensor determines the
reported support. ``max_chain_length`` counts graph edges in this module.
Grouped nodes are disjunctions of at most ``max_group_size`` indicators; strong
partitions exhaust one category's current choices and weak links have a common
exactly-one category. No uniqueness or variant axiom is assumed.

Definitions: HoDoKu tech_chains.php, tech_sdp.php, tech_col.php, tech_wings.php
at https://hodoku.sourceforge.net/en/ and https://www.sudokuwiki.org/3D_Medusa.
"""
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations, islice
from numbers import Integral

from .advanced import _sees, _shared_units, _unit_cells


class _Stop(Exception):
    pass


def _step(scan):
    if not scan.step():
        raise _Stop


@dataclass(frozen=True)
class ImplicationEdge:
    first: frozenset
    second: frozenset
    kind: str
    categories: tuple

    def other(self, node):
        return self.second if node == self.first else self.first


@dataclass
class ImplicationGraph:
    context: object
    scan: object
    nodes: set = field(default_factory=set)
    edges: list = field(default_factory=list)
    adj: dict = field(default_factory=lambda: defaultdict(list))
    truncated: bool = False
    _edge_keys: set = field(default_factory=set)
    output_atoms: set = field(default_factory=set)

    def add(self, first, second, kind, categories):
        if first == second or first & second:
            return
        first, second = frozenset(first), frozenset(second)
        categories = tuple(sorted(categories))
        pair = tuple(sorted((tuple(sorted(first)), tuple(sorted(second)))))
        key = pair, kind, categories
        if key in self._edge_keys:
            return
        limits = self.context.limits
        if len(self.nodes | {first, second}) > limits.max_graph_nodes or len(self.edges) >= limits.max_graph_edges:
            self.truncated = True
            raise _Stop
        self._edge_keys.add(key)
        self.nodes.update((first, second))
        edge = ImplicationEdge(first, second, kind, categories)
        self.edges.append(edge)
        self.adj[first].append(edge)
        self.adj[second].append(edge)


def build_implication_graph(ctx, grouped=False, single_digit=False):
    """Build a bounded graph, exposing original-category provenance per edge.

    The shared ``graph.scan`` also bounds subsequent path queries. A graph cap
    or scan exhaustion sets ``truncated``; partial graphs remain sound but are
    incomplete. Strong groups partition all locally allowed category options.
    Position edges are omitted when ``single_digit=True``.
    """
    graph = ImplicationGraph(ctx, ctx.scan())
    membership = defaultdict(set)
    options = {}
    graph.output_atoms = set().union(*ctx.categories.values())
    try:
        for category, atoms in ctx.categories.items():
            _step(graph.scan)
            current_options = ctx.options(category)
            graph.output_atoms.difference_update(atom for i, atom in enumerate(atoms)
                                                 if i not in current_options)
            if single_digit and category.startswith("pos_"):
                continue
            options[category] = tuple(atoms[i] for i in sorted(current_options))
            for atom in atoms:
                _step(graph.scan)
                membership[atom].add(category)
            current = options[category]
            for atom in current:
                _step(graph.scan)
                node = frozenset((atom,))
                if node not in graph.nodes:
                    if len(graph.nodes) >= ctx.limits.max_graph_nodes:
                        graph.truncated = True
                        raise _Stop
                    graph.nodes.add(node)
            if len(current) == 2:
                graph.add(frozenset(current[:1]), frozenset(current[1:]), "strong", (category,))
            elif grouped and 2 < len(current) <= 2 * ctx.limits.max_group_size:
                first = current[0]
                for size in range(1, min(ctx.limits.max_group_size, len(current) - 1) + 1):
                    for chosen in combinations(current, size):
                        _step(graph.scan)
                        # Each unordered partition is emitted once.
                        if first not in chosen:
                            continue
                        complement = frozenset(current) - set(chosen)
                        if len(complement) <= ctx.limits.max_group_size:
                            graph.add(frozenset(chosen), complement, "strong", (category,))
        contained = defaultdict(list)
        for node in sorted(graph.nodes, key=lambda item: (len(item), tuple(sorted(item)))):
            _step(graph.scan)
            common = set.intersection(*(membership[atom] for atom in node))
            for category in sorted(common):
                contained[category].append(node)
        for category, nodes in contained.items():
            for first, second in combinations(nodes, 2):
                _step(graph.scan)
                if not first & second:
                    graph.add(first, second, "weak", (category,))
    except _Stop:
        graph.truncated = True
    return graph


def implication_path(graph, start_atom, start_truth, end_atom, end_truth,
                     max_length=None, *, forbidden=None):
    """Return a bounded shortest implication proof, or None; charge graph.scan.

    Endpoints can be indicator names or grouped-node frozensets. Exhausting the
    shared work allowance marks the graph truncated. This searches a finite
    implication graph, not assignments or completed Sudoku solutions.
    """
    limit = graph.context.limits.max_chain_length if max_length is None else max_length
    if isinstance(limit, bool) or not isinstance(limit, Integral) or not 1 <= limit <= graph.context.limits.max_chain_length:
        raise ValueError("max_length must be a positive integer <= max_chain_length")
    if not isinstance(start_truth, bool) or not isinstance(end_truth, bool):
        raise ValueError("Implication endpoint truth values must be Boolean")

    def node(value):
        values = (value,) if isinstance(value, str) else tuple(islice(iter(value), graph.context.limits.max_group_size + 1))
        if not values or len(values) > graph.context.limits.max_group_size or any(
                atom not in graph.context.dimensions or not atom.startswith("a_") for atom in values):
            raise ValueError("Implication endpoints must be bounded indicator groups")
        return frozenset(values)

    start, end = node(start_atom), node(end_atom)
    initial = start, bool(start_truth)
    queue = deque([(initial, ())])
    seen = {initial}
    try:
        while queue:
            _step(graph.scan)
            (node, truth), path = queue.popleft()
            if len(path) >= limit:
                continue
            needed = "weak" if truth else "strong"
            for edge in graph.adj.get(node, ()):
                _step(graph.scan)
                if edge.kind != needed or edge == forbidden:
                    continue
                other, following = edge.other(node), not truth
                following_path = (*path, edge)
                if other == end and following == bool(end_truth):
                    return following_path
                state = other, following
                if state not in seen:
                    if not any(candidate.kind == ("weak" if following else "strong")
                               for candidate in graph.adj.get(other, ())):
                        continue
                    seen.add(state)
                    queue.append((state, following_path))
    except _Stop:
        graph.truncated = True
    return None


def describe_implication_paths(ctx, paths, outputs, family="implication", evidence=()):
    """Describe proof unions using all original incidences of their premises."""
    paths = tuple(islice(iter(paths), ctx.limits.max_patterns + 1))
    if len(paths) > ctx.limits.max_patterns:
        raise ValueError("Proof union exceeds max_patterns paths")
    categories = set()
    for path in paths:
        edges = tuple(islice(iter(path), ctx.limits.max_chain_length + 1))
        if len(edges) > ctx.limits.max_chain_length:
            raise ValueError("Proof path exceeds max_chain_length")
        for edge in edges:
            if not isinstance(edge, ImplicationEdge) or edge.kind not in ("strong", "weak"):
                raise ValueError("Expected original-provenance implication edges")
            if not isinstance(edge.categories, (tuple, list)) or not 1 <= len(edge.categories) <= ctx.limits.max_cores:
                raise ValueError("Implication premise categories must be a bounded sequence")
            if any(category not in ctx.categories for category in edge.categories):
                raise ValueError("Unknown implication premise category")
            if not isinstance(edge.first, frozenset) or not isinstance(edge.second, frozenset):
                raise ValueError("Implication groups must be finite frozensets")
            if not edge.first or not edge.second or edge.first & edge.second:
                raise ValueError("Implication groups must be nonempty and disjoint")
            if max(len(edge.first), len(edge.second)) > ctx.limits.max_group_size:
                raise ValueError("Implication group exceeds max_group_size")
            premise_atoms = set().union(*(ctx.categories[category] for category in edge.categories))
            if not (edge.first | edge.second) <= premise_atoms:
                raise ValueError("Implication nodes must belong to original premise categories")
            categories.update(edge.categories)
    outputs = tuple(islice(iter(outputs), ctx.limits.max_graph_nodes + 1))
    if len(outputs) > ctx.limits.max_graph_nodes:
        raise ValueError("Proof outputs exceed max_graph_nodes")
    evidence = tuple(islice(iter(evidence), ctx.limits.max_patterns + 1))
    if len(evidence) > ctx.limits.max_patterns:
        raise ValueError("Proof evidence exceeds max_patterns")
    atoms = {atom for category in categories for atom in ctx.categories[category]}
    return ctx.describe(family, sorted(categories), atoms=sorted(atoms),
                        outputs=tuple(sorted(outputs)), evidence=evidence)


class _Run:
    def __init__(self, ctx, family, scan=None):
        self.ctx, self.family = ctx, family
        self.scan = scan or ctx.scan()
        self.out, self.seen = [], set()
        self.truncated = False

    def step(self):
        _step(self.scan)

    def emit(self, categories, outputs, evidence):
        categories, outputs = tuple(sorted(set(categories))), tuple(sorted(set(outputs)))
        if not outputs:
            return
        key = categories, outputs
        if key in self.seen:
            return
        if len(self.out) >= self.ctx.limits.max_patterns:
            self.truncated = True
            raise _Stop
        atoms = sorted({atom for category in categories for atom in self.ctx.categories[category]})
        self.out.append(self.ctx.describe(self.family, categories, atoms=atoms,
                                          outputs=outputs, evidence=(evidence,)))
        self.seen.add(key)

    def emit_paths(self, paths, outputs, evidence):
        categories = [category for path in paths for edge in path for category in edge.categories]
        self.emit(categories, outputs, evidence)

    def finish(self):
        return self.scan.finish(self.out, truncated=self.truncated)


def _weak_neighbors(graph, node):
    return [(edge.other(node), edge) for edge in graph.adj.get(node, ()) if edge.kind == "weak"]


def detect_aic(ctx, grouped=False, single_digit=False, loops=False):
    """Bounded AIC contradictions/placements, optionally continuous loops.

    Unary outputs retain the original indicator, allowing either false or true
    consequences to emerge solely from support contraction. ``loops=True``
    additionally closes alternating even cycles and projects their weak-link
    common-conflict targets. A weak group link requires one common category.
    """
    family = "grouped_aic" if grouped else "x_chain" if single_digit else "aic"
    return _detect_graph(ctx, family, grouped, single_digit, loops)


def _detect_graph(ctx, family, grouped, single_digit, loops):
    graph = build_implication_graph(ctx, grouped=grouped, single_digit=single_digit)
    run = _Run(ctx, family, graph.scan)
    try:
        if graph.truncated:
            raise _Stop
        for node in sorted(graph.nodes, key=lambda item: (len(item), tuple(sorted(item)))):
            run.step()
            if len(node) != 1:
                continue
            atom = next(iter(node))
            if atom not in graph.output_atoms:
                continue
            for truth in (True, False):
                path = implication_path(graph, atom, truth, atom, not truth)
                if graph.truncated:
                    raise _Stop
                if path:
                    run.emit_paths((path,), (atom,),
                                   f"discontinuous loop; {atom}={int(truth)} implies its complement; edges={len(path)}")
        if loops and ctx.limits.max_chain_length >= 4:
            for closing in graph.edges:
                run.step()
                if closing.kind != "weak":
                    continue
                if not any(edge.kind == "strong" for edge in graph.adj[closing.first]):
                    continue
                path = implication_path(graph, closing.first, False, closing.second, True,
                                        max_length=ctx.limits.max_chain_length - 1,
                                        forbidden=closing)
                if graph.truncated:
                    raise _Stop
                if not path or len(path) < 3:
                    continue
                neighbors = {node: edge for node, edge in _weak_neighbors(graph, closing.first)
                             if len(node) == 1}
                for target, edge in _weak_neighbors(graph, closing.second):
                    run.step()
                    if target in neighbors and not target & (closing.first | closing.second):
                        if not target <= graph.output_atoms:
                            continue
                        run.emit_paths((path, (closing, neighbors[target], edge)), target,
                                       f"continuous alternating loop; edges={len(path) + 1}")
    except _Stop:
        run.truncated = True
    return run.finish()


def detect_x_chains(ctx):
    return detect_aic(ctx, single_digit=True)


def detect_grouped_aic(ctx):
    return detect_aic(ctx, grouped=True)


def detect_nice_loops(ctx, grouped=False, single_digit=False):
    return _detect_graph(ctx, "grouped_nice_loop" if grouped else "nice_loop",
                         grouped, single_digit, loops=True)


def _unit_pairs(ctx, run):
    pairs = []
    for digit in range(ctx.n):
        for kind in ("row", "col", "square"):
            for index in range(ctx.n):
                run.step()
                category = ctx.unit(kind, index, digit)
                choices = sorted(ctx.options(category))
                if len(choices) == 2:
                    cells = _unit_cells(ctx, kind, index)
                    pairs.append((digit, kind, category, cells[choices[0]], cells[choices[1]]))
    return pairs


def _emit_conflict_targets(ctx, run, categories, first, second, digit, excluded, evidence):
    for row in range(ctx.n):
        for col in range(ctx.n):
            run.step()
            target = row, col
            if target in excluded or digit not in ctx.options(ctx.pos(row, col)):
                continue
            if _sees(ctx, target, first) and _sees(ctx, target, second):
                target_categories = (*_shared_units(ctx, target, first, digit),
                                     *_shared_units(ctx, target, second, digit))
                run.emit((*categories, *target_categories), (ctx.atom(row, col, digit),), evidence)


def _detect_turbot(ctx, shape):
    run = _Run(ctx, shape)
    try:
        pairs = _unit_pairs(ctx, run)
        for first, second in combinations(pairs, 2):
            run.step()
            d, akind, acat, a, b = first
            e, bkind, bcat, c, f = second
            if d != e or len({a, b, c, f}) != 4:
                continue
            if shape == "skyscraper" and (akind != bkind or akind not in ("row", "col")):
                continue
            if shape == "two_string_kite" and {akind, bkind} != {"row", "col"}:
                continue
            for outside_a, bridge_a in ((a, b), (b, a)):
                for outside_b, bridge_b in ((c, f), (f, c)):
                    run.step()
                    bridge = _shared_units(ctx, bridge_a, bridge_b, d)
                    if shape == "skyscraper":
                        required = "col_" if akind == "row" else "row_"
                        bridge = [unit for unit in bridge if unit.startswith(required)]
                    if shape == "two_string_kite":
                        bridge = [unit for unit in bridge if unit.startswith("square_")]
                    if not bridge:
                        continue
                    _emit_conflict_targets(ctx, run, (acat, bcat, *bridge),
                                           outside_a, outside_b, d, {a, b, c, f},
                                           f"{shape}; digit={d}; endpoints={outside_a},{outside_b}")
    except _Stop:
        run.truncated = True
    return run.finish()


def detect_turbot_fish(ctx):
    """Two conjugate units joined by a same-digit weak bridge."""
    return _detect_turbot(ctx, "turbot_fish")


def detect_skyscrapers(ctx):
    """Parallel row/column conjugates with a perpendicular-line bridge."""
    return _detect_turbot(ctx, "skyscraper")


def detect_two_string_kites(ctx):
    """One row and one column conjugate with a box bridge."""
    return _detect_turbot(ctx, "two_string_kite")


def detect_w_wings(ctx):
    """Identical bivalue cells linked through an external conjugate pair."""
    run = _Run(ctx, "w_wing")
    try:
        pairs = _unit_pairs(ctx, run)
        bivalue = []
        for row in range(ctx.n):
            for col in range(ctx.n):
                run.step()
                options = ctx.options(ctx.pos(row, col))
                if len(options) == 2:
                    bivalue.append(((row, col), options))
        for (a, digits), (b, other) in combinations(bivalue, 2):
            run.step()
            if digits != other:
                continue
            for x, _, category, first, second in pairs:
                run.step()
                if x not in digits or {a, b} & {first, second}:
                    continue
                y = next(iter(digits - {x}))
                for c, d in ((first, second), (second, first)):
                    run.step()
                    if _sees(ctx, a, c) and _sees(ctx, b, d):
                        categories = (ctx.pos(*a), ctx.pos(*b), category,
                                      *_shared_units(ctx, a, c, x), *_shared_units(ctx, b, d, x))
                        _emit_conflict_targets(ctx, run, categories, a, b, y,
                                               {a, b, first, second},
                                               f"W-Wing; bivalue={a},{b}; conjugate={category}; digit={y}")
    except _Stop:
        run.truncated = True
    return run.finish()


def detect_empty_rectangles(ctx):
    """Box candidates on one row/column cross, plus an external conjugate.

    The cross intersection is absent. Both arms are nonempty and may be groups.
    Transposition and two-candidate crosses are included; each dual instance is
    described separately rather than applying a combined named deduction.
    """
    run = _Run(ctx, "empty_rectangle")
    try:
        pairs = _unit_pairs(ctx, run)
        for digit in range(ctx.n):
            for square in range(ctx.n):
                cells = _unit_cells(ctx, "square", square)
                category = ctx.unit("square", square, digit)
                choices = {cells[index] for index in ctx.options(category)}
                for cross in cells:
                    run.step()
                    r, c = cross
                    if cross in choices or not choices:
                        continue
                    if not all(row == r or col == c for row, col in choices):
                        continue
                    if not any(row == r for row, _ in choices) or not any(col == c for _, col in choices):
                        continue
                    for value, kind, conjugate, first, second in pairs:
                        run.step()
                        if value != digit or kind not in ("row", "col"):
                            continue
                        for near, far in ((first, second), (second, first)):
                            run.step()
                            if kind == "col" and near[0] == r and near[1] // ctx.num != c // ctx.num:
                                target = far[0], c
                                bridge = ctx.unit("row", r, digit)
                                conflict = (ctx.unit("col", c, digit), ctx.unit("row", far[0], digit))
                            elif kind == "row" and near[1] == c and near[0] // ctx.num != r // ctx.num:
                                target = r, far[1]
                                bridge = ctx.unit("col", c, digit)
                                conflict = (ctx.unit("row", r, digit), ctx.unit("col", far[1], digit))
                            else:
                                continue
                            if target not in choices and target not in (first, second) and digit in ctx.options(ctx.pos(*target)):
                                run.emit((category, conjugate, bridge, *conflict),
                                         (ctx.atom(*target, digit),),
                                         f"Empty Rectangle; box={square}; cross={cross}; external={conjugate}")
    except _Stop:
        run.truncated = True
    return run.finish()


def _colored_components(graph, run):
    visited = set()
    components = []
    for start in sorted(graph.nodes, key=lambda node: tuple(sorted(node))):
        run.step()
        if len(start) != 1 or start in visited or not any(edge.kind == "strong" for edge in graph.adj[start]):
            continue
        colors, tree, queue = {start: 0}, [], deque([start])
        odd = []
        while queue:
            run.step()
            node = queue.popleft()
            visited.add(node)
            for edge in graph.adj[node]:
                run.step()
                if edge.kind != "strong":
                    continue
                other = edge.other(node)
                if other not in colors:
                    if len(colors) >= run.ctx.limits.max_coloring_nodes:
                        run.truncated = True
                        raise _Stop
                    colors[other] = 1 - colors[node]
                    tree.append(edge)
                    queue.append(other)
                elif colors[other] == colors[node]:
                    odd.append(edge)
        components.append((colors, tuple(tree), tuple(odd)))
    return components


def detect_medusa(ctx):
    """Bounded multidigit coloring: conflicts, traps, empty cells or units.

    Position bivalue edges and unit conjugates form two-color components.
    Same-color mutual exclusions refute that color. A candidate weakly linked
    to opposite colors is a trap. A color exhausting every current option of
    another exactly-one category is refuted. The latter includes published
    unit-exhaustion extensions of the empty-cell rule. All consequences are
    tensor outputs.
    """
    graph = build_implication_graph(ctx)
    run = _Run(ctx, "medusa", graph.scan)
    try:
        if graph.truncated:
            raise _Stop
        components = _colored_components(graph, run)
        for colors, tree, odd in components:
            if odd:
                run.emit_paths((tree, odd), set().union(*colors), "odd conjugate cycle")
            by_color = ({node for node in colors if colors[node] == 0},
                        {node for node in colors if colors[node] == 1})
            for edge in graph.edges:
                run.step()
                if edge.kind != "weak":
                    continue
                if edge.first in colors and edge.second in colors and colors[edge.first] == colors[edge.second]:
                    outputs = set().union(*by_color[colors[edge.first]])
                    run.emit_paths((tree, (edge,)), outputs, "Medusa same-color conflict")
            for target in sorted(graph.nodes, key=lambda node: tuple(sorted(node))):
                run.step()
                if len(target) != 1 or target in colors:
                    continue
                if not target <= graph.output_atoms:
                    continue
                witnesses = {}
                for neighbor, edge in _weak_neighbors(graph, target):
                    run.step()
                    if neighbor in colors:
                        witnesses.setdefault(colors[neighbor], edge)
                if len(witnesses) == 2:
                    run.emit_paths((tree, tuple(witnesses.values())), target, "Medusa opposite-color trap")
            for category, atoms in ctx.categories.items():
                run.step()
                choices = [frozenset((atoms[index],)) for index in ctx.options(category)]
                if not choices:
                    continue
                for color in (0, 1):
                    witnesses, exhausted = [], True
                    for choice in choices:
                        run.step()
                        if choice in colors and colors[choice] != color:
                            continue
                        witness = None
                        for neighbor, edge in _weak_neighbors(graph, choice):
                            run.step()
                            if neighbor in by_color[color]:
                                witness = edge
                                break
                        if witness is None:
                            exhausted = False
                            break
                        witnesses.append(witness)
                    if exhausted:
                        categories = [cat for edge in (*tree, *witnesses) for cat in edge.categories]
                        run.emit((*categories, category), set().union(*by_color[color]),
                                 f"Medusa empties {category} when color {color} is true")
    except _Stop:
        run.truncated = True
    return run.finish()


def detect_multi_coloring(ctx):
    """Two single-digit conjugate components: both published multi-color cases."""
    graph = build_implication_graph(ctx, single_digit=True)
    run = _Run(ctx, "multi_coloring", graph.scan)
    try:
        if graph.truncated:
            raise _Stop
        components = _colored_components(graph, run)
        weak = {}
        for edge in graph.edges:
            run.step()
            if edge.kind == "weak":
                weak.setdefault(frozenset((edge.first, edge.second)), edge)
        for (ac, atree, _), (bc, btree, _) in combinations(components, 2):
            run.step()
            for a in ac:
                for b in bc:
                    run.step()
                    bridge = weak.get(frozenset((a, b)))
                    if bridge is None:
                        continue
                    # Type 1: the colors opposite a,b contain an at-least-one
                    # true alternative. Keep every target's original conflicts.
                    for target in sorted(graph.nodes, key=lambda node: tuple(sorted(node))):
                        run.step()
                        if len(target) != 1 or target in ac or target in bc:
                            continue
                        if not target <= graph.output_atoms:
                            continue
                        wa, wb = None, None
                        for neighbor, edge in _weak_neighbors(graph, target):
                            run.step()
                            if neighbor in ac and ac[neighbor] != ac[a]:
                                wa = edge
                            if neighbor in bc and bc[neighbor] != bc[b]:
                                wb = edge
                        if wa and wb:
                            run.emit_paths((atree, btree, (bridge, wa, wb)), target,
                                           "Multi-colors type 1: weak component bridge")
            # Type 2, both orientations: one color conflicts with both colors
            # of the other component, so that entire color is false.
            for first, ftree, second, stree in ((ac, atree, bc, btree), (bc, btree, ac, atree)):
                for color in (0, 1):
                    witnesses = {}
                    for node in first:
                        run.step()
                        if first[node] != color:
                            continue
                        for neighbor, edge in _weak_neighbors(graph, node):
                            run.step()
                            if neighbor in second:
                                witnesses.setdefault(second[neighbor], edge)
                    if len(witnesses) == 2:
                        outputs = set().union(*(node for node in first if first[node] == color))
                        run.emit_paths((ftree, stree, tuple(witnesses.values())), outputs,
                                       "Multi-colors type 2: color conflicts with both opposite colors")
    except _Stop:
        run.truncated = True
    return run.finish()


__all__ = ["ImplicationEdge", "ImplicationGraph", "build_implication_graph",
           "implication_path", "describe_implication_paths", "detect_aic",
           "detect_x_chains", "detect_grouped_aic", "detect_nice_loops",
           "detect_turbot_fish", "detect_skyscrapers", "detect_two_string_kites",
           "detect_empty_rectangles", "detect_w_wings", "detect_medusa",
           "detect_multi_coloring"]
