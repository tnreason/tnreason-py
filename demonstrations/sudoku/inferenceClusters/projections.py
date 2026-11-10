"""Exact template, overlay and finite forcing-proof cluster descriptions.

Sources: https://hodoku.sourceforge.net/en/tech_last.php . These selectors
construct original-tensor proof regions, not solutions or hand-coded removals.
A region is a proposal until exact contraction reports supported output values.
"""
from collections import defaultdict, deque
from collections.abc import Mapping
from numbers import Integral

from .core import RuleCluster


def _finite(values, label, maximum):
    if not isinstance(values, (tuple, list, set, frozenset, range, dict)):
        raise TypeError(f"{label} must be a finite collection")
    if len(values) > maximum:
        raise ValueError(f"{label} exceeds its configured bound")
    return tuple(sorted(values, key=repr)) if isinstance(values, (set, frozenset)) else tuple(values)


def _digits(ctx, digits):
    digits = tuple(range(ctx.n)) if digits is None else _finite(digits, "digits", ctx.n)
    if len(set(digits)) != len(digits) or any(isinstance(d, bool) or not isinstance(d, Integral)
                                           or not 0 <= d < ctx.n for d in digits):
        raise ValueError("digits must be distinct zero-based digits")
    return digits


def describe_templates(ctx, digit, *, outputs=None):
    """One digit's complete row/column/box incidence network.

    Projecting each row category gives all surviving placements, including
    both eliminations and forced placements. This exactly implements the
    template support relation without enumerating templates or solving Sudoku.
    """
    digit, = _digits(ctx, (digit,))
    categories = tuple(ctx.unit(kind, index, digit)
                       for kind in ("row", "col", "square") for index in range(ctx.n))
    if outputs is None:
        outputs = tuple(ctx.unit("row", index, digit) for index in range(ctx.n))
    return ctx.describe("templates", categories, outputs=outputs,
                        evidence=("complete single-digit incidence", digit))


def detect_templates(ctx, *, digits=None):
    scan, out = ctx.scan(), []
    for digit in _digits(ctx, digits):
        for _ in range(3 * ctx.n):
            if not scan.step():
                return scan.finish(out, truncated=True)
        out.append(describe_templates(ctx, digit))
        if len(out) >= ctx.limits.max_patterns:
            return scan.finish(out, truncated=True)
    return scan.finish(out)


def describe_pattern_overlay(ctx, digits, *, outputs=None):
    """Couple complete single-digit template networks by cell constraints.

    All original links of the chosen digits participate. A cell category
    enforces that two selected digits cannot occupy the same cell; unselected
    digits remain available under their original domain restrictions.
    """
    digits = _digits(ctx, digits)
    if len(digits) < 2:
        raise ValueError("Pattern overlay requires at least two digits")
    categories = tuple(ctx.unit(kind, index, d) for d in digits
                       for kind in ("row", "col", "square") for index in range(ctx.n))
    categories += tuple(ctx.pos(r, c) for r in range(ctx.n) for c in range(ctx.n))
    if outputs is None:
        outputs = tuple(ctx.unit("row", index, d) for d in digits for index in range(ctx.n))
    atoms = tuple(ctx.atom(r, c, d) for d in digits for r in range(ctx.n) for c in range(ctx.n))
    return ctx.describe("pattern_overlay", categories, atoms=atoms, outputs=outputs,
                        evidence=("coupled complete templates", tuple(digits)))


def detect_pattern_overlay(ctx, *, digits=None, outputs=None):
    """Describe one caller-chosen digit overlay, default all ordinary digits.

    A full overlay may exceed the contraction budgets. It is reported as such,
    never silently replaced by a subset of digit templates.
    """
    digits = _digits(ctx, digits)
    scan = ctx.scan()
    for _ in range(3 * ctx.n * len(digits) + ctx.n ** 2):
        if not scan.step():
            return scan.finish((), truncated=True)
    return scan.finish((describe_pattern_overlay(ctx, digits, outputs=outputs),))


def combine_clusters(ctx, family, clusters, *, outputs=None, evidence=()):
    """Union original proof tensors; never multiply derived summaries.

    Combining descriptions does not reuse another context's domain-dependent
    result. All inputs are freshly restricted by this context at execution.
    """
    clusters = _finite(clusters, "clusters", ctx.limits.max_patterns)
    if not clusters or any(not isinstance(c, RuleCluster) for c in clusters):
        raise ValueError("Provide at least one RuleCluster")
    if any(c.assumptions for c in clusters):
        raise ValueError("Unencoded proof assumptions are unsupported")
    keys = tuple(dict.fromkeys(k for c in clusters for k in c.core_keys))
    if outputs is None:
        outputs = tuple(dict.fromkeys(o for c in clusters for o in c.open_colors))
    return ctx.describe_tensors(family, keys, outputs, evidence=evidence)


def describe_forcing_net(ctx, source, branches, *, outputs):
    """Join an explicit finite proof for every alternative of a source.

    ``branches`` maps every current source value to a RuleCluster (or a finite
    sequence of clusters). Its keys must exhaust the source domain. Each proof
    must name original tensors; branch assumptions or contracted summaries are
    not accepted. The union's exact projection proves the common consequence.
    It can also refute a candidate by exhibiting no support, without search.
    """
    if source not in ctx.categories:
        raise ValueError("The forcing source must be an original categorical variable")
    if not isinstance(branches, Mapping) or len(branches) > ctx.limits.max_petals:
        raise ValueError("Provide bounded source-value -> original proof branches")
    expected = set(ctx.options(source))
    if not expected or set(branches) != expected:
        raise ValueError("Proof branches must exhaust every current source alternative")
    proofs = []
    for value in sorted(branches):
        branch = branches[value]
        if isinstance(branch, RuleCluster):
            proofs.append(branch)
        else:
            proofs.extend(_finite(branch, "branch proofs", ctx.limits.max_patterns))
        if len(proofs) > ctx.limits.max_patterns:
            raise ValueError("Too many branch proof clusters")
    # All source incidence links encode the exhaustive categorical alternative,
    # including any alternatives not mentioned in the smaller proof paths.
    proofs.append(ctx.describe("forcing_source", (source,), atoms=ctx.categories[source],
                               outputs=(source,)))
    if len(proofs) > ctx.limits.max_patterns:
        raise ValueError("Too many branch proof clusters including source")
    return combine_clusters(ctx, "forcing_net", tuple(proofs), outputs=outputs,
                            evidence=("exhaustive source alternatives", source,
                                      tuple(sorted(expected))))


def detect_proof_regions(ctx, *, seeds=None, radius=None):
    """Bounded connected original proof regions around candidate sources.

    This is structural selection, not a claim that each region forces a value.
    It includes branched dependencies in the region, so exact projection can
    establish consequences unavailable to individual-link GAC. The region's
    scope/radius is explicit; there is no candidate assignment, recursive CSP
    search, branch propagation, or completed-solution lookup.
    """
    radius = min(2, ctx.limits.max_net_radius) if radius is None else radius
    if isinstance(radius, bool) or not isinstance(radius, Integral) or not 1 <= radius <= ctx.limits.max_net_radius:
        raise ValueError("radius must lie within max_net_radius")
    if seeds is not None:
        seeds = _finite(seeds, "seeds", len(ctx.categories))
        if set(seeds) - ctx.categories.keys():
            raise ValueError("Unknown categorical source")
    scan, out, seen, by_atom, allowed = ctx.scan(), [], set(), defaultdict(list), {}
    for category, atoms in ctx.categories.items():
        if not scan.step():
            return scan.finish(out, truncated=True)
        allowed[category] = ctx.options(category)
        for value in allowed[category]:
            if not scan.step():
                return scan.finish(out, truncated=True)
            by_atom[atoms[value]].append(category)
    if seeds is None:
        seeds = tuple(c for c in ctx.categories if 1 < len(allowed[c]) <= ctx.limits.max_petals)
    truncated = False
    for source in seeds:
        if not scan.step():
            return scan.finish(out, truncated=True)
        if not 1 < len(allowed[source]) <= ctx.limits.max_petals:
            continue
        selected, queue = {source}, deque(((source, 0),))
        while queue:
            category, depth = queue.popleft()
            if depth >= radius:
                continue
            for value in sorted(allowed[category]):
                atom = ctx.categories[category][value]
                for neighbor in by_atom[atom]:
                    if not scan.step():
                        return scan.finish(out, truncated=True)
                    if neighbor in selected:
                        continue
                    if len(selected) >= ctx.limits.max_net_categories:
                        truncated = True
                        continue
                    selected.add(neighbor)
                    queue.append((neighbor, depth + 1))
        if len(selected) < 2:
            continue
        # The source's complete original incidence makes source alternatives
        # explicit even when the neighborhood cap cuts a dependency edge.
        counts = defaultdict(int)
        for category in selected:
            for atom in ctx.categories[category]:
                counts[atom] += 1
        atoms = set(ctx.categories[source]) | {a for a, count in counts.items() if count >= 2}
        cluster = ctx.describe("forcing_net", selected, atoms=atoms, outputs=(source,),
                               evidence=("bounded original proof region", source, radius,
                                         len(selected), ctx.limits.max_net_categories))
        signature = cluster.core_keys, cluster.open_colors
        if signature not in seen:
            seen.add(signature)
            out.append(cluster)
            if len(out) >= ctx.limits.max_patterns:
                return scan.finish(out, truncated=True)
    return scan.finish(out, truncated=truncated)


def detect_nishio(ctx, *, digits=None):
    """Single-digit consistency projection via the exact template relation.

    A zero-supported placement refutes that candidate; this performs no trial
    assignment or search. The original single-digit tensors are the proof.
    """
    detection = detect_templates(ctx, digits=digits)
    clusters = tuple(ctx.describe_tensors("nishio", c.core_keys, c.open_colors,
                                          evidence=("single-digit contradiction support", *c.evidence))
                     for c in detection.clusters)
    return type(detection)(clusters, detection.steps, detection.truncated)


def describe_complete_support(ctx, *, outputs=None):
    """The complete original network, with explicit bounded execution.

    Every ordinary deduction from the encoded finite constraints is a support
    consequence of this relation. This is a completeness-in-principle fallback,
    not a promise that a large puzzle's contraction will fit any finite budget.
    Additional original variant/arithmetic tensors participate unchanged.
    """
    if outputs is None:
        outputs = tuple(ctx.pos(r, c) for r in range(ctx.n) for c in range(ctx.n))
    return ctx.describe_tensors("complete_support", tuple(ctx.cores), outputs,
                                evidence=("all original tensors", len(ctx.cores)))


def detect_complete_support(ctx, *, outputs=None):
    scan = ctx.scan()
    for _ in ctx.cores:
        if not scan.step():
            return scan.finish((), truncated=True)
    return scan.finish((describe_complete_support(ctx, outputs=outputs),))
