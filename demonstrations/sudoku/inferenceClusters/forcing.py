"""Bounded exact-one forcing proofs and compound Kraken fish descriptions.

Tracing recognizes common consequences, contradictions and finite implication
paths. It never applies them to the network. Every returned cluster contains
only original incidence constraints; exact Boolean contraction supplies the
actual supported output values. Assumptions used to trace a source alternative
are never installed as constraints or domain masks.

Source: https://hodoku.sourceforge.net/en/tech_last.php . Nets allow converging
false premises of an original exact-one category; chains use only original
conjugate pairs. There is no recursive case splitting or completed-grid search.
"""
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

from .core import RuleCluster
from .fish_general import _Stop as _FishStop, _cells, _motifs, _validate
from .implication import build_implication_graph, describe_implication_paths, implication_path
from .projections import combine_clusters


class _Stop(Exception):
    pass


def _step(scan):
    if not scan.step():
        raise _Stop


def _finite(values, label, maximum):
    if not isinstance(values, (tuple, list, set, frozenset)):
        raise TypeError(label + " must be a finite collection")
    if len(values) > maximum:
        raise ValueError(label + " exceeds its configured bound")
    return tuple(sorted(values)) if isinstance(values, (set, frozenset)) else tuple(values)


@dataclass(frozen=True)
class _Proof:
    categories: frozenset
    depth: int


@dataclass
class _Branch:
    literals: dict
    contradiction: _Proof | None = None
    complete: bool = True


def _trace(ctx, scan, source_atom, active, membership, nets):
    """Propagate one symbolic premise, retaining original-category proofs."""
    literals, queue, premises = {}, deque(), set()
    initial = _Proof(frozenset(), 0)
    literals[source_atom] = True, initial
    queue.append(source_atom)

    def derive(atom, truth, categories, depth):
        nonlocal premises
        if atom in literals and literals[atom][0] == truth:
            return None
        if (depth > ctx.limits.max_chain_length
                or len(premises | categories) > ctx.limits.max_net_categories):
            return _Branch(literals, complete=False)
        proof = _Proof(frozenset(categories), depth)
        if atom in literals:
            known, previous = literals[atom]
            if known != truth:
                combined = previous.categories | proof.categories
                if len(combined) > ctx.limits.max_net_categories:
                    return _Branch(literals, complete=False)
                return _Branch(literals, _Proof(combined, max(previous.depth, depth)))
            return None
        if len(literals) >= ctx.limits.max_graph_nodes:
            return _Branch(literals, complete=False)
        premises.update(categories)
        literals[atom] = truth, proof
        queue.append(atom)
        return None

    while queue:
        _step(scan)
        atom = queue.popleft()
        truth, proof = literals[atom]
        for category in membership.get(atom, ()):
            _step(scan)
            options = active[category]
            if truth:
                for other in options:
                    _step(scan)
                    if other == atom:
                        continue
                    result = derive(other, False, proof.categories | {category}, proof.depth + 1)
                    if result is not None:
                        return result
            elif nets or len(options) == 2:
                false_proofs = [literals[other][1] for other in options
                                if other in literals and literals[other][0] is False]
                remaining = tuple(other for other in options
                                  if other not in literals or literals[other][0] is True)
                if len(remaining) > 1:
                    continue
                categories = frozenset((category,)).union(*(p.categories for p in false_proofs))
                depth = 1 + max((p.depth for p in false_proofs), default=0)
                if not remaining:
                    if depth > ctx.limits.max_chain_length or len(categories) > ctx.limits.max_net_categories:
                        return _Branch(literals, complete=False)
                    return _Branch(literals, _Proof(categories, depth))
                result = derive(remaining[0], True, categories, depth)
                if result is not None:
                    return result
    return _Branch(literals)


def _forcing(ctx, sources, nets):
    if sources is not None:
        sources = _finite(sources, "sources", len(ctx.categories))
        if set(sources) - ctx.categories.keys():
            raise ValueError("Unknown original forcing source category")
    scan, active, membership = ctx.scan(), {}, defaultdict(list)
    out, seen, truncated = [], set(), False
    family = "forcing_net" if nets else "forcing_chain"

    def emit(categories, outputs, evidence):
        nonlocal truncated
        if len(categories) > ctx.limits.max_net_categories:
            truncated = True
            return
        categories = tuple(sorted(categories))
        outputs = tuple(sorted(outputs))
        signature = categories, outputs
        if signature in seen:
            return
        seen.add(signature)
        atoms = tuple(sorted({atom for category in categories for atom in ctx.categories[category]}))
        out.append(ctx.describe(family, categories, atoms=atoms, outputs=outputs, evidence=evidence))
        if len(out) >= ctx.limits.max_patterns:
            raise _Stop

    try:
        for category, atoms in ctx.categories.items():
            _step(scan)
            options = ctx.options(category)
            active[category] = tuple(atoms[index] for index in sorted(options))
            for atom in active[category]:
                _step(scan)
                if atom not in membership and len(membership) >= ctx.limits.max_graph_nodes:
                    return scan.finish(out, truncated=True)
                membership[atom].append(category)
        if sources is None:
            sources = tuple(category for category, options in active.items()
                            if 1 < len(options) <= ctx.limits.max_petals or not options)
        for source in sources:
            _step(scan)
            options = active[source]
            if not options:
                emit({source}, (source,), (("source", source), ("contradiction", "empty source")))
                continue
            if not 1 < len(options) <= ctx.limits.max_petals:
                continue
            branches = []
            for atom in options:
                _step(scan)
                branches.append(_trace(ctx, scan, atom, active, membership, nets))
            if any(not branch.complete for branch in branches):
                truncated = True
                continue
            failures = [branch.contradiction for branch in branches if branch.contradiction is not None]
            survivors = [branch for branch in branches if branch.contradiction is None]
            failure_categories = frozenset().union(*(proof.categories for proof in failures))
            if not survivors:
                emit({source} | set(failure_categories), (source,),
                     (("source", source), ("alternatives", options), ("contradiction", "all alternatives refuted")))
                continue
            common = set(survivors[0].literals)
            for branch in survivors[1:]:
                common &= branch.literals.keys()
            for atom in sorted(common):
                _step(scan)
                truth = survivors[0].literals[atom][0]
                if (any(branch.literals[atom][0] != truth for branch in survivors)
                        or ctx.domains[atom] == (int(truth),)):
                    continue
                categories = {source} | set(failure_categories)
                for branch in survivors:
                    categories.update(branch.literals[atom][1].categories)
                emit(categories, (atom,), (("source", source), ("alternatives", options),
                                         ("refuted", len(failures)), ("consequence", (atom, truth)),
                                         ("mode", "net" if nets else "chain")))
    except _Stop:
        return scan.finish(out, truncated=True)
    return scan.finish(out, truncated=truncated)


def detect_forcing_chains(ctx, *, sources=None):
    """Common truths/refuted cases from exhaustive original source alternatives."""
    return _forcing(ctx, sources, nets=False)


def detect_forcing_nets(ctx, *, sources=None):
    """Exact-one proof nets, including convergence of several false premises.

    Branch event, premise, source-alternative and implication-depth bounds are
    ``max_graph_nodes``, ``max_net_categories``, ``max_petals`` and
    ``max_chain_length``. An incomplete branch makes that source unknown;
    partial tracing never silently refutes a source alternative.
    """
    return _forcing(ctx, sources, nets=True)


def _fish_metadata(ctx, fish):
    if not isinstance(fish, RuleCluster) or fish.assumptions:
        raise ValueError("Provide an original fish cluster without extra assumptions")
    try:
        data = dict(fish.evidence)
        digit, bases, covers = data["digit"], data["bases"], data["covers"]
        _, _, bases, covers, _ = _validate(ctx, (len(bases),), digit, bases, covers, "all", None)
        if set(bases) & set(covers):
            raise ValueError("Fish bases and covers must be distinct houses")
        declared_fins = []
        for label in ("exo_fins", "endo_fins"):
            cells = _finite(data[label], label, ctx.n ** 2)
            if any(not isinstance(cell, (tuple, list)) or len(cell) != 2
                   or any(isinstance(value, bool) or not isinstance(value, Integral)
                          or not 0 <= value < ctx.n for value in cell) for cell in cells):
                raise ValueError("Fish fins must be bounded zero-based cell pairs")
            declared_fins.append(set(map(tuple, cells)))
        allowed = {}
        for house in bases:
            cells = _cells(ctx, house)
            allowed[house] = {cells[value] for value in ctx.options(ctx.unit(*house, digit))}
        candidates = set().union(*allowed.values())
        covered = set().union(*(set(_cells(ctx, house)) for house in covers))
        exo = candidates - covered
        endo = {cell for cell in candidates if sum(cell in allowed[house] for house in bases) > 1}
        if declared_fins != [exo, endo]:
            raise ValueError("Fish fin metadata does not match this exact domain snapshot")
    except (KeyError, TypeError) as exc:
        raise ValueError("Fish metadata must identify digit, bases, covers, exo_fins and endo_fins") from exc
    # Metadata is not a substitute for the original counting premises. Repair
    # a caller's smaller fish description by explicitly including every base
    # and cover incidence, while preserving any additional original tensors.
    categories = tuple(ctx.unit(*house, digit) for house in (*bases, *covers))
    atoms = tuple(sorted({atom for category in categories for atom in ctx.categories[category]}))
    body = ctx.describe("kraken_required_body", categories, atoms=atoms, outputs=())
    keys = tuple(dict.fromkeys((*fish.core_keys, *body.core_keys)))
    complete = ctx.describe_tensors("kraken_fish_body", keys, fish.open_colors, evidence=fish.evidence)
    return digit, bases, covers, candidates, exo | endo, complete


def _compound(ctx, fish, proofs, expected, outputs, family, evidence):
    if not isinstance(proofs, Mapping):
        raise ValueError("Provide a bounded candidate-to-proof mapping")
    if len(proofs) + 1 > ctx.limits.max_patterns:
        raise ValueError("Compound proof count exceeds max_patterns")
    if set(proofs) != set(expected):
        raise ValueError("Proofs must exhaust every required fin/base candidate")
    outputs = _finite(outputs, "outputs", ctx.limits.max_graph_nodes)
    if not outputs or set(outputs) - ctx.dimensions.keys():
        raise ValueError("Provide known retained output variables")
    for atom, proof in proofs.items():
        if (not isinstance(proof, RuleCluster) or proof.assumptions
                or atom not in proof.participating_variables
                or set(outputs) - set(proof.participating_variables)):
            raise ValueError("Every proof must connect its premise and all retained outputs using original tensors")
    return combine_clusters(ctx, family, (fish, *proofs.values()), outputs=outputs, evidence=evidence)


def describe_kraken_fish(ctx, fish, fin_proofs, *, outputs):
    """Type 1: union a fish body with an original proof for every exo/endo fin.

    Proofs are ``fin atom -> RuleCluster``. Their exact tensor union, rather
    than a caller's stated implication, determines whether an output is forced.
    Fish metadata is validated against the current context's domains.
    """
    digit, _, _, _, fins, complete = _fish_metadata(ctx, fish)
    if not fins:
        raise ValueError("Kraken type 1 requires at least one exo or endo fin")
    expected = {ctx.atom(*cell, digit) for cell in fins}
    return _compound(ctx, complete, fin_proofs, expected, outputs, "kraken_fish",
                     (("type", 1), ("fins", tuple(sorted(expected)))))


def describe_kraken_type2(ctx, fish, cover, case_proofs, *, outputs):
    """Type 2: cover every base candidate in one cover house and every fin."""
    digit, _, covers, candidates, fins, complete = _fish_metadata(ctx, fish)
    if not isinstance(cover, (tuple, list)) or len(cover) != 2:
        raise ValueError("The selected cover must be a finite house pair")
    cover = tuple(cover)
    if cover not in covers:
        raise ValueError("The selected type-2 house must be a fish cover")
    expected_cells = (candidates & set(_cells(ctx, cover))) | fins
    expected = {ctx.atom(*cell, digit) for cell in expected_cells}
    if not expected:
        raise ValueError("The selected cover has no base/fin premise")
    return _compound(ctx, complete, case_proofs, expected, outputs, "kraken_type2",
                     (("type", 2), ("cover", cover), ("cases", tuple(sorted(expected)))))


def _body(ctx, motif):
    categories = tuple(ctx.unit(*house, motif.digit) for house in (*motif.bases, *motif.covers))
    atoms = tuple(sorted({atom for category in categories for atom in ctx.categories[category]}))
    return ctx.describe("kraken_fish_body", categories, atoms=atoms, outputs=(),
                        evidence=(("digit", motif.digit), ("bases", motif.bases),
                                  ("covers", motif.covers), ("exo_fins", motif.exo),
                                  ("endo_fins", motif.endo)))


def detect_kraken_fish(ctx, sizes=None, *, digit=None, bases=None, covers=None,
                       targets=None, grouped=False, kind=1, max_fins=None):
    """Fish plus bounded implication proofs for type 1 or type 2 conclusions.

    ``targets`` is an optional finite collection of candidate indicator names.
    Type 1 considers covered non-base fish-digit targets; type 2 also permits
    other digits and requires a chain from every base candidate in one cover
    plus every fin. Ordinary and grouped graph links are available. Explicit
    bases/covers use the generalized fish house-selection format.
    """
    if kind not in (1, 2) or isinstance(kind, bool):
        raise ValueError("Kraken kind must be 1 or 2")
    if not isinstance(grouped, bool):
        raise TypeError("grouped must be Boolean")
    sizes, digits, bases, covers, max_fins = _validate(ctx, sizes, digit, bases, covers, "all", max_fins)
    if targets is not None:
        targets = _finite(targets, "targets", ctx.limits.max_graph_nodes)
        if any(atom not in ctx.dimensions or not atom.startswith("a_") for atom in targets):
            raise ValueError("Kraken targets must be original Boolean indicators")
    graph = build_implication_graph(ctx, grouped=grouped)
    out, seen, pattern_seen = [], set(), set()
    if graph.truncated:
        return graph.scan.finish(out, truncated=True)
    try:
        motifs = _motifs(ctx, graph.scan, sizes, digits, bases, covers, "all", True, True,
                         max_fins, require_visibility=False, patterns_only=(kind == 2))
        for motif in motifs:
            _step(graph.scan)
            fins = tuple(ctx.atom(*cell, motif.digit) for cell in sorted(set(motif.exo) | set(motif.endo)))
            if kind == 1:
                if not fins or motif.cannibalistic:
                    continue
                target_atoms = (ctx.atom(*motif.target, motif.digit),)
                if targets is not None and target_atoms[0] not in targets:
                    continue
                premise_sets = ((None, fins),)
            else:
                signature = motif.digit, motif.bases, motif.covers
                if signature in pattern_seen:
                    continue
                pattern_seen.add(signature)
                target_atoms = tuple(sorted(graph.output_atoms)) if targets is None else targets
                data = _fish_metadata(ctx, _body(ctx, motif))
                candidates, fin_cells = data[3], data[4]
                premise_sets = tuple((cover, tuple(ctx.atom(*cell, motif.digit) for cell in sorted(
                    (candidates & set(_cells(ctx, cover))) | fin_cells))) for cover in motif.covers)
            for target in target_atoms:
                _step(graph.scan)
                if ctx.domains[target] != (0, 1):
                    continue
                for cover, premises in premise_sets:
                    _step(graph.scan)
                    if not premises or target in premises or len(premises) + 1 > ctx.limits.max_patterns:
                        continue
                    proofs = {}
                    for premise in premises:
                        _step(graph.scan)
                        path = implication_path(graph, premise, True, target, False)
                        if graph.truncated:
                            raise _Stop
                        if path is None:
                            break
                        proofs[premise] = describe_implication_paths(
                            ctx, (path,), outputs=(target,), family="kraken_case",
                            evidence=(("premise", premise), ("consequence", (target, False))))
                    if len(proofs) != len(premises):
                        continue
                    fish = _body(ctx, motif)
                    cluster = (describe_kraken_fish(ctx, fish, proofs, outputs=(target,)) if kind == 1
                               else describe_kraken_type2(ctx, fish, cover, proofs, outputs=(target,)))
                    key = cluster.core_keys, cluster.open_colors
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(cluster)
                    if len(out) >= ctx.limits.max_patterns:
                        return graph.scan.finish(out, truncated=True)
    except (_Stop, _FishStop):
        return graph.scan.finish(out, truncated=True)
    return graph.scan.finish(out, truncated=graph.truncated)


def detect_kraken_type2(ctx, sizes=None, **options):
    return detect_kraken_fish(ctx, sizes, kind=2, **options)


__all__ = ["detect_forcing_chains", "detect_forcing_nets", "describe_kraken_fish",
           "describe_kraken_type2", "detect_kraken_fish", "detect_kraken_type2"]
