"""Bounded mixed-house, multiply finned, endo and cannibalistic fish.

The counting premise is HoDoKu's general fish definition: base candidates in
several bases are endo fins, uncovered base candidates are exo fins, and a
potential elimination must see every fin.  A covered candidate outside all
bases is an ordinary target; a base candidate in several covers is a
cannibalistic target.  All conclusions are obtained from selected original
links by ``contract_cluster``, never from this counting argument.

References: https://hodoku.sourceforge.net/en/tech_fishg.php and
https://hodoku.sourceforge.net/en/tech_fishc.php.  A completed bounded scan is
not a claim of unlimited fish coverage.  Explicit ``bases``/``covers`` and
``cover_pairs`` make particular large-board motifs independently selectable.
"""
from dataclasses import dataclass
from itertools import combinations
from numbers import Integral


_KINDS = ("row", "col", "square")
_FAMILIES = {"all", "basic", "franken", "mutant", "multi_fin", "endo", "cannibalistic"}


class _Stop(Exception):
    pass


def _step(scan):
    if not scan.step():
        raise _Stop


def _sizes(ctx, requested):
    maximum = min(ctx.n, ctx.limits.max_fish_size)
    if requested is None:
        return tuple(range(2, maximum + 1))
    if not isinstance(requested, (tuple, list, range)):
        raise TypeError("sizes must be a finite tuple, list, or range")
    if len(requested) > ctx.limits.max_fish_size:
        raise ValueError("Too many requested fish sizes")
    if any(isinstance(value, bool) or not isinstance(value, Integral)
           or not 2 <= value <= maximum for value in requested):
        raise ValueError("Fish size exceeds max_fish_size or board size")
    return tuple(sorted(set(map(int, requested))))


def _houses(ctx, selected):
    if selected is None:
        return None
    if not isinstance(selected, (tuple, list)):
        raise TypeError("House selections must be finite tuples or lists")
    if not 2 <= len(selected) <= min(ctx.n, ctx.limits.max_fish_size):
        raise ValueError("House selection exceeds the fish size bound")
    result = []
    for house in selected:
        if (not isinstance(house, (tuple, list)) or len(house) != 2
                or house[0] not in _KINDS or isinstance(house[1], bool)
                or not isinstance(house[1], Integral) or not 0 <= house[1] < ctx.n):
            raise ValueError("A house must be (row/col/square, zero-based index)")
        result.append((house[0], int(house[1])))
    if len(set(result)) != len(result):
        raise ValueError("A house selection cannot contain duplicates")
    return tuple(sorted(result, key=lambda house: (_KINDS.index(house[0]), house[1])))


def _cells(ctx, house):
    kind, index = house
    if kind == "row":
        return tuple((index, col) for col in range(ctx.n))
    if kind == "col":
        return tuple((row, index) for row in range(ctx.n))
    return tuple((index // ctx.num * ctx.num + offset // ctx.num,
                  index % ctx.num * ctx.num + offset % ctx.num)
                 for offset in range(ctx.n))


def _conflicts(ctx, first, second):
    """Original houses proving that two distinct candidate cells conflict."""
    if first == second:
        return ()
    houses = []
    if first[0] == second[0]:
        houses.append(("row", first[0]))
    if first[1] == second[1]:
        houses.append(("col", first[1]))
    if (first[0] // ctx.num, first[1] // ctx.num) == (second[0] // ctx.num, second[1] // ctx.num):
        houses.append(("square", first[0] // ctx.num * ctx.num + first[1] // ctx.num))
    return tuple(houses)


def _taxonomy(bases, covers):
    a, b = {kind for kind, _ in bases}, {kind for kind, _ in covers}
    if (a, b) in (({"row"}, {"col"}), ({"col"}, {"row"})):
        return "basic"
    if "square" in a | b and (
            (a <= {"row", "square"} and b <= {"col", "square"})
            or (a <= {"col", "square"} and b <= {"row", "square"})):
        return "franken"
    return "mutant"


@dataclass(frozen=True)
class _Motif:
    digit: int
    bases: tuple
    covers: tuple
    taxonomy: str
    exo: tuple
    endo: tuple
    target: tuple
    cannibalistic: bool
    categories: tuple


def _validate(ctx, sizes, digit, bases, covers, family, max_fins):
    if family not in _FAMILIES:
        raise ValueError("Unknown generalized fish family")
    sizes = _sizes(ctx, sizes)
    bases, covers = _houses(ctx, bases), _houses(ctx, covers)
    if bases is not None:
        if len(bases) not in sizes:
            raise ValueError("Explicit base count is not among requested sizes")
    if covers is not None:
        if len(covers) not in sizes:
            raise ValueError("Explicit cover count is not among requested sizes")
        if bases is not None and len(bases) != len(covers):
            raise ValueError("Base and cover counts must be equal")
    if digit is not None and (isinstance(digit, bool) or not isinstance(digit, Integral)
                              or not 0 <= digit < ctx.n):
        raise ValueError("digit must be a zero-based Sudoku digit")
    if max_fins is None:
        max_fins = ctx.n ** 2
    if isinstance(max_fins, bool) or not isinstance(max_fins, Integral) or not 0 <= max_fins <= ctx.n ** 2:
        raise ValueError("max_fins must lie between zero and the number of cells")
    return sizes, (range(ctx.n) if digit is None else (int(digit),)), bases, covers, int(max_fins)


def _motifs(ctx, scan, sizes, digits, explicit_bases, explicit_covers, family,
            fins, endo, max_fins, cover_pairs=None, require_visibility=True,
            patterns_only=False):
    houses = tuple((kind, index) for kind in _KINDS for index in range(ctx.n))
    geometry = {house: frozenset(_cells(ctx, house)) for house in houses}
    for digit in digits:
        allowed = {}
        for house in houses:
            _step(scan)
            cells = _cells(ctx, house)
            unit = ctx.unit(*house, digit)
            allowed[house] = frozenset(cells[value] for value in ctx.options(unit))
        for size in sizes:
            if explicit_bases is not None and len(explicit_bases) != size:
                continue
            base_selections = (explicit_bases,) if explicit_bases is not None else combinations(houses, size)
            for bases in base_selections:
                _step(scan)
                if any(not allowed[house] for house in bases):
                    continue
                base_candidates = frozenset().union(*(allowed[house] for house in bases))
                base_counts = {cell: sum(cell in allowed[house] for house in bases) for cell in base_candidates}
                endo_cells = frozenset(cell for cell, count in base_counts.items() if count > 1)
                if endo_cells and not endo:
                    continue
                if cover_pairs is not None:
                    cover_selections = cover_pairs
                elif explicit_covers is not None:
                    cover_selections = (explicit_covers,) if len(explicit_covers) == size else ()
                else:
                    pool = tuple(house for house in houses if house not in bases)
                    cover_selections = combinations(pool, size)
                for covers in cover_selections:
                    _step(scan)
                    if len(covers) != size or set(bases) & set(covers):
                        continue
                    taxonomy = _taxonomy(bases, covers)
                    if family in {"basic", "franken", "mutant"} and family != taxonomy:
                        continue
                    covered = frozenset().union(*(geometry[house] for house in covers))
                    exo_cells = base_candidates - covered
                    fin_cells = exo_cells | endo_cells
                    if (len(fin_cells) > max_fins or (fin_cells and not fins)
                            or (family == "endo" and not endo_cells)):
                        continue
                    fin_boxes = {(row // ctx.num, col // ctx.num) for row, col in fin_cells}
                    if family == "multi_fin" and len(fin_boxes) < 2:
                        continue
                    # Degenerate bodies are intentional: if fin-false leaves
                    # a base empty, the same exhaustive fin premise still
                    # proves a target seeing every fin false.
                    if patterns_only:
                        categories = tuple(sorted(ctx.unit(*house, digit) for house in (*bases, *covers)))
                        yield _Motif(digit, tuple(bases), tuple(covers), taxonomy,
                                     tuple(sorted(exo_cells)), tuple(sorted(endo_cells)),
                                     None, False, categories)
                        continue
                    for target in sorted(covered):
                        _step(scan)
                        if target in fin_cells or digit not in ctx.options(ctx.pos(*target)):
                            continue
                        in_base = target in base_candidates
                        multiplicity = sum(target in geometry[house] for house in covers)
                        cannibalistic = in_base and multiplicity > 1
                        if in_base and not cannibalistic:
                            continue
                        if family == "cannibalistic" and not cannibalistic:
                            continue
                        conflict_houses = set()
                        sees_all = True
                        for fin in sorted(fin_cells) if require_visibility else ():
                            _step(scan)
                            shared = _conflicts(ctx, target, fin)
                            if not shared:
                                sees_all = False
                                break
                            conflict_houses.update(shared)
                        if not sees_all:
                            continue
                        categories = tuple(sorted({ctx.unit(*house, digit)
                                                   for house in (*bases, *covers, *conflict_houses)}
                                                  | {ctx.pos(*target)}))
                        yield _Motif(digit, tuple(bases), tuple(covers), taxonomy,
                                     tuple(sorted(exo_cells)), tuple(sorted(endo_cells)),
                                     target, cannibalistic, categories)


def _description(ctx, motif, family):
    evidence = (("digit", motif.digit), ("bases", motif.bases), ("covers", motif.covers),
                ("taxonomy", motif.taxonomy), ("exo_fins", motif.exo),
                ("endo_fins", motif.endo), ("cannibalistic", motif.cannibalistic),
                ("target", motif.target))
    return ctx.describe(family, motif.categories, outputs=(ctx.pos(*motif.target),), evidence=evidence)


def detect_general_fish(ctx, sizes=None, *, family="all", digit=None, bases=None,
                        covers=None, fins=True, endo=True, max_fins=None):
    """Select arbitrary row/column/box fish with optional exo and endo fins.

    ``bases`` and ``covers`` are optional exact finite house selections, e.g.
    ``(("row", 0), ("square", 3))``. Otherwise selections are enumerated within
    ``max_steps``. ``family`` can restrict taxonomy or require multi-box fins,
    endo fins, or a cannibalistic target. ``max_fins`` bounds the total number
    of distinct exo/endo fin cells; default is the finite board cell count.

    Every participating base and cover is distinct. Overlapping base candidate
    cells are treated as fins; overlapping cover membership is retained and
    permits cannibalism. All original target/fin conflict-house links enter the
    cluster, including when different fins are seen through different houses.
    """
    sizes, digits, bases, covers, max_fins = _validate(ctx, sizes, digit, bases, covers, family, max_fins)
    if not isinstance(fins, bool) or not isinstance(endo, bool):
        raise TypeError("fins and endo must be Boolean")
    scan, out, seen = ctx.scan(), [], set()
    try:
        for motif in _motifs(ctx, scan, sizes, digits, bases, covers, family, fins, endo, max_fins):
            signature = motif.categories, motif.target
            if signature in seen:
                continue
            seen.add(signature)
            name = "general_fish" if family == "all" else family + "_fish"
            out.append(_description(ctx, motif, name))
            if len(out) >= ctx.limits.max_patterns:
                return scan.finish(out, truncated=True)
    except _Stop:
        return scan.finish(out, truncated=True)
    return scan.finish(out)


def detect_franken_fish(ctx, sizes=None, **options):
    return detect_general_fish(ctx, sizes, family="franken", **options)


def detect_mutant_fish(ctx, sizes=None, **options):
    return detect_general_fish(ctx, sizes, family="mutant", **options)


def detect_multi_fin_fish(ctx, sizes=None, **options):
    return detect_general_fish(ctx, sizes, family="multi_fin", **options)


def detect_endo_fish(ctx, sizes=None, **options):
    return detect_general_fish(ctx, sizes, family="endo", **options)


def detect_cannibalistic_fish(ctx, sizes=None, **options):
    return detect_general_fish(ctx, sizes, family="cannibalistic", **options)


def detect_siamese_fish(ctx, sizes=None, *, digit=None, bases=None, cover_pairs=None,
                        max_fins=None, max_components=None):
    """Join two finned fish sharing bases and differing in one cover house.

    Components must have the same fish taxonomy and distinct target cells.
    ``cover_pairs`` can provide two exact cover selections. Otherwise bounded
    enumeration retains at most ``max_components`` component descriptions
    (default twice ``max_patterns``) and compares them within the same scan
    budget. A composite contains the union of original component categories
    and projects both targets by exact contraction.
    """
    sizes, digits, bases, _, max_fins = _validate(ctx, sizes, digit, bases, None, "all", max_fins)
    if cover_pairs is not None:
        if not isinstance(cover_pairs, (tuple, list)) or len(cover_pairs) != 2:
            raise ValueError("cover_pairs must contain exactly two finite cover selections")
        cover_pairs = tuple(_houses(ctx, selected) for selected in cover_pairs)
        if (len(cover_pairs[0]) != len(cover_pairs[1]) or len(cover_pairs[0]) not in sizes
                or len(set(cover_pairs[0]) ^ set(cover_pairs[1])) != 2):
            raise ValueError("Siamese covers must have equal sizes and differ in one house")
    if max_components is None:
        max_components = min(2 * ctx.limits.max_patterns, ctx.limits.max_steps)
    if (isinstance(max_components, bool) or not isinstance(max_components, Integral)
            or not 1 <= max_components <= ctx.limits.max_steps):
        raise ValueError("max_components must be between 1 and max_steps")
    scan, out, components, seen = ctx.scan(), [], [], set()
    try:
        for motif in _motifs(ctx, scan, sizes, digits, bases, None, "all", True, True,
                             max_fins, cover_pairs=cover_pairs):
            if not (motif.exo or motif.endo):
                continue
            for previous in components:
                _step(scan)
                if (motif.digit != previous.digit or motif.bases != previous.bases
                        or motif.taxonomy != previous.taxonomy or motif.target == previous.target
                        or len(set(motif.covers) ^ set(previous.covers)) != 2):
                    continue
                outputs = tuple(sorted({ctx.pos(*motif.target), ctx.pos(*previous.target)}))
                categories = tuple(sorted(set(motif.categories) | set(previous.categories)))
                signature = categories, outputs
                if signature in seen:
                    continue
                seen.add(signature)
                out.append(ctx.describe("siamese_fish", categories, outputs=outputs,
                                        evidence=(("digit", motif.digit), ("bases", motif.bases),
                                                  ("covers", (previous.covers, motif.covers)),
                                                  ("targets", (previous.target, motif.target)))))
                if len(out) >= ctx.limits.max_patterns:
                    return scan.finish(out, truncated=True)
            if len(components) >= max_components:
                return scan.finish(out, truncated=True)
            components.append(motif)
    except _Stop:
        return scan.finish(out, truncated=True)
    return scan.finish(out)


__all__ = ["detect_general_fish", "detect_franken_fish", "detect_mutant_fish",
           "detect_multi_fin_fish", "detect_endo_fish", "detect_cannibalistic_fish",
           "detect_siamese_fish"]
