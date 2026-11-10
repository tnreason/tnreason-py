"""Bounded ALS and counting motifs, described using original Sudoku links.

The selectors do not implement candidate removals.  They select original
position/indicator/house incidence tensors; ``contract_cluster`` performs every
deduction.  A selected motif may therefore produce no additional restriction.
Every automatic scan shares max_steps/max_patterns, and its intermediate set
catalogue is capped by max_als_sets.  max_chain_length counts ALS nodes here;
max_petals bounds stem alternatives and aligned-exclusion helper sets.

Definitions and scope:
https://hodoku.sourceforge.net/en/tech_als.php (overlap, double links, chains,
Death Blossom), https://hodoku.sourceforge.net/en/tech_misc.php (Sue de Coq),
https://www.sudokuwiki.org/WXYZ_wing (bent sets), and
https://www.sudokuwiki.org/Aligned_Pair_Exclusion (aligned exclusion).
"""

from dataclasses import dataclass
from itertools import combinations
from numbers import Integral

from .advanced import _sees, _unit_cells


class _Stop(Exception):
    pass


class _Enumeration:
    def __init__(self, ctx):
        self.ctx, self.scan = ctx, ctx.scan()
        self.out, self.seen, self.truncated = [], set(), False

    def step(self):
        if not self.scan.step():
            raise _Stop

    def emit(self, cluster):
        signature = cluster.family, cluster.core_keys, cluster.open_colors
        if signature in self.seen:
            return
        if len(self.out) >= self.ctx.limits.max_patterns:
            raise _Stop
        self.seen.add(signature)
        self.out.append(cluster)

    def finish(self, stopped=False):
        return self.scan.finish(self.out, truncated=stopped or self.truncated)


@dataclass(frozen=True)
class AlmostLockedSet:
    """Inspectable ALS description; pass through ``make_als`` to validate it."""
    cells: tuple
    digits: frozenset
    house: tuple


def _cell(ctx, cell):
    if (not isinstance(cell, (tuple, list)) or len(cell) != 2
            or any(isinstance(x, bool) or not isinstance(x, Integral)
                   or not 0 <= x < ctx.n for x in cell)):
        raise ValueError("Cells must be in-board (row, column) integer pairs")
    return tuple(map(int, cell))


def _cells(ctx, cells, maximum):
    if isinstance(cells, AlmostLockedSet):
        cells = cells.cells
    if not isinstance(cells, (tuple, list, set, frozenset)):
        raise TypeError("Cell selections must be finite explicit collections")
    if not 1 <= len(cells) <= maximum:
        raise ValueError("Cell selection exceeds its configured size bound")
    result = tuple(sorted(_cell(ctx, cell) for cell in cells))
    if len(set(result)) != len(result):
        raise ValueError("Cell selections must contain distinct cells")
    return result


def _digit(ctx, digit):
    if (isinstance(digit, bool) or not isinstance(digit, Integral)
            or not 0 <= digit < ctx.n):
        raise ValueError("Digit must be a zero-based in-board integer")
    return int(digit)


def _houses(ctx, cells):
    rows, cols = {c[0] for c in cells}, {c[1] for c in cells}
    boxes = {c[0] // ctx.num * ctx.num + c[1] // ctx.num for c in cells}
    return tuple((kind, next(iter(indices))) for kind, indices in
                 (("row", rows), ("col", cols), ("square", boxes)) if len(indices) == 1)


def _options(ctx, cells):
    return frozenset().union(*(ctx.options(ctx.pos(*cell)) for cell in cells))


def make_als(ctx, cells, house=None):
    """Validate k unsolved cells in a house with exactly k+1 current digits.

    ``house`` is an optional ("row"|"col"|"square", zero-based index).
    Explicit descriptors and automatic selectors use the same validation.
    """
    if isinstance(cells, AlmostLockedSet):
        if house is None:
            house = cells.house
        cells = cells.cells
    cells = _cells(ctx, cells, ctx.limits.max_als_size)
    houses = _houses(ctx, cells)
    if house is not None and (not isinstance(house, (tuple, list)) or len(house) != 2):
        raise ValueError("ALS house must be an explicit finite (kind, index) pair")
    if not houses or (house is not None and tuple(house) not in houses):
        raise ValueError("An ALS must be contained in the declared ordinary house")
    if any(len(ctx.options(ctx.pos(*cell))) < 2 for cell in cells):
        raise ValueError("An ALS requires unsolved cells")
    digits = _options(ctx, cells)
    if len(digits) != len(cells) + 1:
        raise ValueError("An ALS needs exactly k+1 digits in k cells")
    return AlmostLockedSet(cells, digits, houses[0] if house is None else tuple(house))


def _board(ctx, e):
    result = {}
    for row in range(ctx.n):
        for col in range(ctx.n):
            e.step()
            result[row, col] = ctx.options(ctx.pos(row, col))
    return result


def _catalogue(ctx, e, candidates, excesses=(1,)):
    """Finite catalogue of house-contained almost/near-locked cell sets."""
    result, seen = [], set()
    for kind in ("row", "col", "square"):
        for index in range(ctx.n):
            e.step()
            cells = tuple(cell for cell in _unit_cells(ctx, kind, index)
                          if 2 <= len(candidates[cell])
                          <= ctx.limits.max_als_size + max(excesses))
            for size in range(1, min(ctx.limits.max_als_size, len(cells)) + 1):
                for chosen in combinations(cells, size):
                    e.step()
                    chosen = tuple(sorted(chosen))
                    if chosen in seen:
                        continue
                    digits = frozenset().union(*(candidates[cell] for cell in chosen))
                    if len(digits) - size not in excesses:
                        continue
                    if len(result) >= ctx.limits.max_als_sets:
                        e.truncated = True
                        return tuple(result)
                    seen.add(chosen)
                    result.append(AlmostLockedSet(chosen, digits, (kind, index)))
    return tuple(result)


def _occurrences(ctx, als, digit):
    return tuple(cell for cell in als.cells if digit in ctx.options(ctx.pos(*cell)))


def _restricted(ctx, first, second, digit, e=None):
    if digit not in first.digits & second.digits:
        return False
    # An overlapping cell must not itself contain the RCC.
    if any(digit in ctx.options(ctx.pos(*cell))
           for cell in set(first.cells) & set(second.cells)):
        return False
    for a in _occurrences(ctx, first, digit):
        for b in _occurrences(ctx, second, digit):
            if e is not None:
                e.step()
            if not _sees(ctx, a, b):
                return False
    return True


def _sees_all(ctx, target, sources):
    return bool(sources) and all(_sees(ctx, target, source) for source in sources)


def _motif(ctx, family, cells, outputs, evidence):
    """Select all original ordinary-house links incident to current motif atoms.

    Explicit atoms avoid dragging intersections outside the chosen motif into
    the description.  For each involved cell and each of its current digits,
    include its position and all three house incidences.  Thus original tensors
    enforce every same-digit conflict in the motif.  Domain-free projection is
    still sound, but can be weaker because absent current digits were not used
    to select house links.
    """
    cells = tuple(sorted(set(cells)))
    categories = {ctx.pos(*cell) for cell in cells}
    atoms = []
    for row, col in cells:
        for digit in sorted(ctx.options(ctx.pos(row, col))):
            atoms.append(ctx.atom(row, col, digit))
            categories.update((ctx.unit("row", row, digit),
                               ctx.unit("col", col, digit),
                               ctx.unit("square", row // ctx.num * ctx.num + col // ctx.num, digit)))
    if not atoms:
        raise ValueError("Motif contains no current candidate indicators")
    return ctx.describe(family, sorted(categories), atoms=atoms,
                        outputs=tuple(sorted({ctx.pos(*cell) for cell in outputs})),
                        evidence=evidence)


def describe_als_xz(ctx, first, second, x, z, target):
    """Describe ordinary ALS-XZ, allowing overlap away from RCC occurrences."""
    first, second = make_als(ctx, first), make_als(ctx, second)
    x, z, target = _digit(ctx, x), _digit(ctx, z), _cell(ctx, target)
    if first.cells == second.cells or not _restricted(ctx, first, second, x):
        raise ValueError("ALS-XZ requires two different ALSs and a valid RCC")
    if z == x or z not in first.digits & second.digits:
        raise ValueError("ALS-XZ needs a common non-RCC elimination digit")
    sources = _occurrences(ctx, first, z) + _occurrences(ctx, second, z)
    if (target in set(first.cells) | set(second.cells)
            or z not in ctx.options(ctx.pos(*target)) or not _sees_all(ctx, target, sources)):
        raise ValueError("ALS-XZ target must see every common-digit occurrence")
    return _motif(ctx, "als_xz_extended", (*first.cells, *second.cells, target),
                  (target,), ("ALS-XZ", first.cells, second.cells, x, z, target))


def describe_doubly_linked_als(ctx, first, second, x, y, digit, target):
    """Describe RCC-union or individual non-RCC locking from two RCCs.

    Non-RCC targets may belong to the other ALS (cannibalistic elimination),
    but cannot belong to the ALS whose digit is locked.  RCC targets are
    external and must see all occurrences across both ALSs.
    """
    first, second = make_als(ctx, first), make_als(ctx, second)
    x, y, digit = _digit(ctx, x), _digit(ctx, y), _digit(ctx, digit)
    target = _cell(ctx, target)
    if (x == y or first.cells == second.cells or not _restricted(ctx, first, second, x)
            or not _restricted(ctx, first, second, y)):
        raise ValueError("Doubly linked ALSs require two distinct valid RCCs")
    if digit not in ctx.options(ctx.pos(*target)):
        raise ValueError("Target candidate is absent")
    if digit in (x, y):
        sources = _occurrences(ctx, first, digit) + _occurrences(ctx, second, digit)
        valid = target not in set(first.cells) | set(second.cells) and _sees_all(ctx, target, sources)
    else:
        valid = any(digit in als.digits and target not in als.cells
                    and _sees_all(ctx, target, _occurrences(ctx, als, digit))
                    for als in (first, second))
    if not valid:
        raise ValueError("Target is not an RCC-union or non-RCC locked-set consequence")
    return _motif(ctx, "als_doubly_linked", (*first.cells, *second.cells, target),
                  (target,), ("double RCC", first.cells, second.cells, x, y, digit, target))


def detect_als_xz_extended(ctx, kind="both", allow_overlap=True):
    """Enumerate singly/doubly linked ALSs under the configured finite caps."""
    if kind not in ("singly", "doubly", "both"):
        raise ValueError("kind must be singly, doubly, or both")
    e = _Enumeration(ctx)
    try:
        candidates = _board(ctx, e)
        sets = _catalogue(ctx, e, candidates)
        for first, second in combinations(sets, 2):
            e.step()
            if not allow_overlap and set(first.cells) & set(second.cells):
                continue
            rccs = []
            for digit in sorted(first.digits & second.digits):
                e.step()
                if _restricted(ctx, first, second, digit, e):
                    rccs.append(digit)
            if kind != "doubly":
                for x in rccs:
                    for z in sorted((first.digits & second.digits) - {x}):
                        sources = _occurrences(ctx, first, z) + _occurrences(ctx, second, z)
                        for target, options in candidates.items():
                            e.step()
                            if (z in options and target not in set(first.cells) | set(second.cells)
                                    and _sees_all(ctx, target, sources)):
                                e.emit(describe_als_xz(ctx, first, second, x, z, target))
            if kind != "singly":
                for x, y in combinations(rccs, 2):
                    for digit in sorted(first.digits | second.digits):
                        for target, options in candidates.items():
                            e.step()
                            if digit not in options:
                                continue
                            if digit in (x, y):
                                sources = (_occurrences(ctx, first, digit)
                                           + _occurrences(ctx, second, digit))
                                valid = (target not in set(first.cells) | set(second.cells)
                                         and _sees_all(ctx, target, sources))
                            else:
                                valid = any(digit in als.digits and target not in als.cells
                                            and _sees_all(ctx, target, _occurrences(ctx, als, digit))
                                            for als in (first, second))
                            if valid:
                                e.emit(describe_doubly_linked_als(ctx, first, second,
                                                                x, y, digit, target))
    except _Stop:
        return e.finish(True)
    return e.finish()


def describe_als_chain(ctx, sets, rccs, digit, target, *, family="als_chain"):
    """Describe a finite ALS chain with distinct adjacent RCCs.

    Overlap is allowed when each joining RCC is absent from that overlap.
    The endpoint digit differs from each endpoint's incident RCC.  More complex
    doubly linked chains are safe to project too, but no shortcut inference
    replaces contraction.
    """
    if (not isinstance(sets, (tuple, list)) or not 2 <= len(sets) <= ctx.limits.max_chain_length
            or not isinstance(rccs, (tuple, list)) or len(rccs) != len(sets) - 1):
        raise ValueError("ALS chain needs bounded explicit sets and one RCC per edge")
    sets = tuple(make_als(ctx, als) for als in sets)
    if len({als.cells for als in sets}) != len(sets):
        raise ValueError("An ALS chain cannot repeat an ALS node")
    rccs = tuple(_digit(ctx, digit) for digit in rccs)
    digit, target = _digit(ctx, digit), _cell(ctx, target)
    if any(a == b for a, b in zip(rccs, rccs[1:])):
        raise ValueError("Adjacent ALS chain RCCs must be different")
    if any(not _restricted(ctx, a, b, link) for a, b, link in zip(sets, sets[1:], rccs)):
        raise ValueError("Invalid ALS chain RCC")
    if digit not in sets[0].digits & sets[-1].digits or digit in (rccs[0], rccs[-1]):
        raise ValueError("ALS chain endpoint digit must differ from endpoint RCCs")
    cells = set().union(*(set(als.cells) for als in sets))
    sources = _occurrences(ctx, sets[0], digit) + _occurrences(ctx, sets[-1], digit)
    if target in cells or digit not in ctx.options(ctx.pos(*target)) or not _sees_all(ctx, target, sources):
        raise ValueError("ALS chain target must see both endpoint digit groups")
    return _motif(ctx, family, (*cells, target), (target,),
                  ("ALS chain", tuple(als.cells for als in sets), rccs, digit, target))


def _detect_chains(ctx, lengths, family):
    default_lengths = lengths is None
    if default_lengths:
        if ctx.limits.max_chain_length < 3:
            return ctx.scan().finish(())
    else:
        if (not isinstance(lengths, (tuple, list, set, frozenset)) or not lengths
                or any(isinstance(size, bool) or not isinstance(size, Integral)
                       or not 2 <= size <= ctx.limits.max_chain_length for size in lengths)):
            raise ValueError("Chain lengths must be a finite selection within max_chain_length")
        lengths = frozenset(size for size in lengths if size <= ctx.limits.max_als_sets)
        if not lengths:
            return ctx.scan().finish(())
    e = _Enumeration(ctx)
    try:
        candidates = _board(ctx, e)
        sets = _catalogue(ctx, e, candidates)
        # No default range is materialized before the bounded catalogue exists.
        # A path cannot contain more distinct nodes than this actual catalogue.
        if default_lengths:
            maximum = min(ctx.limits.max_chain_length, len(sets))
            lengths = range(3, maximum + 1)
        else:
            lengths = frozenset(size for size in lengths if size <= len(sets))
            maximum = max(lengths, default=0)
        if not lengths:
            return e.finish()
        graph = {i: [] for i in range(len(sets))}
        for i, j in combinations(range(len(sets)), 2):
            e.step()
            for digit in sorted(sets[i].digits & sets[j].digits):
                e.step()
                if _restricted(ctx, sets[i], sets[j], digit, e):
                    graph[i].append((j, digit))
                    graph[j].append((i, digit))

        def walk(path, links):
            e.step()
            if len(path) in lengths and path[0] < path[-1]:
                first, last = sets[path[0]], sets[path[-1]]
                excluded = set().union(*(set(sets[index].cells) for index in path))
                for digit in sorted((first.digits & last.digits) - {links[0], links[-1]}):
                    sources = _occurrences(ctx, first, digit) + _occurrences(ctx, last, digit)
                    for target, options in candidates.items():
                        e.step()
                        if (digit in options and target not in excluded
                                and _sees_all(ctx, target, sources)):
                            e.emit(describe_als_chain(ctx, [sets[index] for index in path],
                                                     links, digit, target, family=family))
            if len(path) >= maximum:
                return
            for neighbor, digit in graph[path[-1]]:
                e.step()
                if neighbor not in path and (not links or digit != links[-1]):
                    walk((*path, neighbor), (*links, digit))

        for start in graph:
            walk((start,), ())
    except _Stop:
        return e.finish(True)
    return e.finish()


def detect_als_chains(ctx, lengths=None):
    return _detect_chains(ctx, lengths, "als_chain")


def detect_als_xy_wing(ctx):
    """Three ALS nodes, two distinct RCCs, and a common endpoint digit."""
    return _detect_chains(ctx, (3,), "als_xy_wing")


def describe_death_blossom(ctx, stem, petals, digit, target):
    """Validate one ALS petal for every stem alternative; petals may overlap.

    ``petals`` maps each stem digit to an explicit ALS.  When the common digit
    also occurs in the stem, the target must additionally see the stem itself.
    """
    stem, target, digit = _cell(ctx, stem), _cell(ctx, target), _digit(ctx, digit)
    alternatives = ctx.options(ctx.pos(*stem))
    if (not isinstance(petals, dict) or not 2 <= len(alternatives) <= ctx.limits.max_petals
            or set(petals) != set(alternatives)):
        raise ValueError("Death Blossom needs one bounded petal per stem alternative")
    petals = {value: make_als(ctx, als) for value, als in petals.items()}
    sources, cells = [], {stem}
    for value, als in petals.items():
        if (stem in als.cells or value not in als.digits or digit not in als.digits
                or not _sees_all(ctx, stem, _occurrences(ctx, als, value))):
            raise ValueError("Invalid stem/petal RCC or common petal digit")
        cells.update(als.cells)
        sources.extend(_occurrences(ctx, als, digit))
    if digit in alternatives:
        sources.append(stem)
    if target in cells or digit not in ctx.options(ctx.pos(*target)) or not _sees_all(ctx, target, sources):
        raise ValueError("Death Blossom target does not see all common-digit sources")
    return _motif(ctx, "death_blossom", (*cells, target), (target,),
                  ("Death Blossom", stem, tuple((v, als.cells) for v, als in sorted(petals.items())),
                   digit, target))


def detect_death_blossom(ctx):
    e = _Enumeration(ctx)
    try:
        candidates = _board(ctx, e)
        sets = _catalogue(ctx, e, candidates)
        for stem, alternatives in candidates.items():
            e.step()
            if not 2 <= len(alternatives) <= ctx.limits.max_petals:
                continue
            digits = tuple(sorted(alternatives))
            choices = []
            for value in digits:
                possible = []
                for als in sets:
                    e.step()
                    if (stem not in als.cells and value in als.digits
                            and _sees_all(ctx, stem, _occurrences(ctx, als, value))):
                        possible.append(als)
                choices.append(possible)

            def choose(index, petals, common):
                e.step()
                if index == len(digits):
                    cells = {stem} | set().union(*(set(als.cells) for als in petals.values()))
                    for digit in sorted(common):
                        sources = tuple(cell for als in petals.values()
                                        for cell in _occurrences(ctx, als, digit))
                        if digit in alternatives:
                            sources += (stem,)
                        for target, options in candidates.items():
                            e.step()
                            if (digit in options and target not in cells
                                    and _sees_all(ctx, target, sources)):
                                e.emit(describe_death_blossom(ctx, stem, petals, digit, target))
                    return
                for als in choices[index]:
                    e.step()
                    shared = als.digits if common is None else common & als.digits
                    if shared:
                        choose(index + 1, {**petals, digits[index]: als}, shared)

            if all(choices):
                choose(0, {}, None)
    except _Stop:
        return e.finish(True)
    return e.finish()


def _nonrestricted(ctx, cells, digits, candidates, e=None):
    result = []
    for digit in sorted(digits):
        if e is not None:
            e.step()
        sources = tuple(cell for cell in cells if digit in candidates[cell])
        for a, b in combinations(sources, 2):
            if e is not None:
                e.step()
            if not _sees(ctx, a, b):
                result.append(digit)
                break
    return tuple(result)


def describe_bent_subset(ctx, cells, digit, target):
    """k cells/k digits with precisely one nonrestricted digit (bent subset)."""
    cells = _cells(ctx, cells, ctx.limits.max_subset_size)
    digit, target = _digit(ctx, digit), _cell(ctx, target)
    candidates = {cell: ctx.options(ctx.pos(*cell)) for cell in cells}
    digits = frozenset().union(*candidates.values())
    if (len(cells) < 3 or len(digits) != len(cells)
            or any(len(options) < 2 for options in candidates.values())
            or _nonrestricted(ctx, cells, digits, candidates) != (digit,)):
        raise ValueError("Bent subset needs equal cells/digits and one nonrestricted digit")
    sources = tuple(cell for cell in cells if digit in candidates[cell])
    if target in cells or digit not in ctx.options(ctx.pos(*target)) or not _sees_all(ctx, target, sources):
        raise ValueError("Bent-subset target must see every nonrestricted-digit occurrence")
    return _motif(ctx, "bent_subset", (*cells, target), (target,),
                  ("bent subset", cells, digit, target))


def detect_bent_subsets(ctx, sizes=None):
    """Bounded generalized bent triples/quads and larger configured subsets.

    The all-but-one restricted-digit counting condition does not require a
    particular hinge cell.  The WXYZ wrapper additionally requires two-house
    coverage, matching the published generalized four-cell definition.
    """
    return _detect_bent(ctx, sizes, False)


def _two_houses_cover(ctx, cells, e):
    houses = []
    for kind in ("row", "col", "square"):
        for index in range(ctx.n):
            e.step()
            houses.append(set(_unit_cells(ctx, kind, index)))
    selected = set(cells)
    for a, b in combinations(houses, 2):
        e.step()
        if selected <= a | b:
            return True
    return False


def _detect_bent(ctx, sizes, two_houses):
    if sizes is None:
        sizes = tuple(range(3, min(ctx.limits.max_subset_size, ctx.n) + 1))
        if not sizes:
            return ctx.scan().finish(())
    if (not isinstance(sizes, (tuple, list, set, frozenset)) or not sizes
            or any(isinstance(size, bool) or not isinstance(size, Integral)
                   or not 3 <= size <= ctx.limits.max_subset_size for size in sizes)):
        raise ValueError("Bent sizes must be finite integers within max_subset_size")
    e = _Enumeration(ctx)
    try:
        candidates = _board(ctx, e)
        for size in sorted(set(size for size in sizes if size <= ctx.n)):
            eligible = tuple(cell for cell, options in candidates.items() if 2 <= len(options) <= size)
            for cells in combinations(eligible, size):
                e.step()
                digits = frozenset().union(*(candidates[cell] for cell in cells))
                if len(digits) != size:
                    continue
                nonrestricted = _nonrestricted(ctx, cells, digits, candidates, e)
                if len(nonrestricted) != 1 or (two_houses and not _two_houses_cover(ctx, cells, e)):
                    continue
                digit = nonrestricted[0]
                sources = tuple(cell for cell in cells if digit in candidates[cell])
                for target, options in candidates.items():
                    e.step()
                    if digit in options and target not in cells and _sees_all(ctx, target, sources):
                        cluster = describe_bent_subset(ctx, cells, digit, target)
                        if two_houses:
                            cluster = _motif(ctx, "wxyz_wing", (*cells, target), (target,), cluster.evidence)
                        e.emit(cluster)
    except _Stop:
        return e.finish(True)
    return e.finish()


def detect_wxyz_wings(ctx):
    return _detect_bent(ctx, (4,), True)


def _sdc_sets(ctx, intersection, box_cells, line_cells, line, box):
    intersection = _cells(ctx, intersection, min(ctx.num, ctx.limits.max_subset_size))
    box_cells = _cells(ctx, box_cells, ctx.limits.max_als_size)
    line_cells = _cells(ctx, line_cells, ctx.limits.max_als_size)
    if (not isinstance(line, (tuple, list)) or len(line) != 2
            or line[0] not in ("row", "col")
            or isinstance(line[1], bool) or not isinstance(line[1], Integral)
            or not 0 <= line[1] < ctx.n
            or isinstance(box, bool) or not isinstance(box, Integral) or not 0 <= box < ctx.n):
        raise ValueError("Sue de Coq requires an ordinary line and box")
    ic, bc, rc = set(intersection), set(box_cells), set(line_cells)
    bh, rh = set(_unit_cells(ctx, "square", box)), set(_unit_cells(ctx, *line))
    if (len(ic) < 2 or not ic <= bh & rh or not bc <= bh or not rc <= rh
            or ic & bc or ic & rc or bc & rc):
        raise ValueError("Sue de Coq cell sectors must be disjoint and in their houses")
    if any(len(ctx.options(ctx.pos(*cell))) < 2 for cell in ic | bc | rc):
        raise ValueError("Sue de Coq requires unsolved cells")
    v, vb, vr = _options(ctx, ic), _options(ctx, bc), _options(ctx, rc)
    eb, er = len(vb) - len(bc), len(vr) - len(rc)
    if (len(v) < len(ic) + 2 or v & vb & vr or eb < 1 or er < 1
            or len(ic) != eb + er + len(v - vb - vr)):
        raise ValueError("Sue de Coq disjoint-sector counting conditions do not hold")
    return intersection, box_cells, line_cells, v, vb, vr, bh, rh


def describe_sue_de_coq(ctx, intersection, box_cells, line_cells, line, box, digit, target):
    """Validate basic or generalized two-sector disjoint subset counting."""
    ic, bc, rc, v, vb, vr, bh, rh = _sdc_sets(ctx, intersection, box_cells, line_cells, line, box)
    digit, target = _digit(ctx, digit), _cell(ctx, target)
    box_valid = target in bh - set(ic) - set(bc) and digit in vb | (v - vr)
    line_valid = target in rh - set(ic) - set(rc) and digit in vr | (v - vb)
    if digit not in ctx.options(ctx.pos(*target)) or not (box_valid or line_valid):
        raise ValueError("Target is not a Sue de Coq sector consequence")
    return _motif(ctx, "sue_de_coq", (*ic, *bc, *rc, target), (target,),
                  ("Sue de Coq", ic, bc, rc, tuple(line), box, digit, target))


def detect_sue_de_coq(ctx, extended=True):
    e = _Enumeration(ctx)
    try:
        candidates = _board(ctx, e)
        for kind in ("row", "col"):
            for index in range(ctx.n):
                row_cells = set(_unit_cells(ctx, kind, index))
                for box in range(ctx.n):
                    e.step()
                    box_cells = set(_unit_cells(ctx, "square", box))
                    cross = tuple(sorted(cell for cell in row_cells & box_cells
                                         if len(candidates[cell]) >= 2))
                    for count in range(2, min(len(cross), ctx.limits.max_subset_size) + 1):
                        for intersection in combinations(cross, count):
                            e.step()
                            v = frozenset().union(*(candidates[cell] for cell in intersection))
                            if len(v) < count + 2 or (not extended and len(v) != count + 2):
                                continue

                            def sides(house):
                                choices = tuple(sorted(cell for cell in house - set(intersection)
                                                       if len(candidates[cell]) >= 2))
                                result = []
                                cap = ctx.limits.max_als_size if extended else 1
                                for size in range(1, min(cap, len(choices)) + 1):
                                    for chosen in combinations(choices, size):
                                        e.step()
                                        digits = frozenset().union(*(candidates[cell] for cell in chosen))
                                        if len(digits) <= size or (not extended and not (len(digits) == 2 and digits <= v)):
                                            continue
                                        if len(result) >= ctx.limits.max_als_sets:
                                            e.truncated = True
                                            return result
                                        result.append((chosen, digits))
                                return result

                            bs, rs = sides(box_cells), sides(row_cells)
                            for (bc, vb), (rc, vr) in ((a, b) for a in bs for b in rs):
                                e.step()
                                if (set(bc) & set(rc) or v & vb & vr
                                        or count != len(vb) - len(bc) + len(vr) - len(rc) + len(v - vb - vr)):
                                    continue
                                for target in sorted(box_cells | row_cells):
                                    for digit in sorted(candidates[target]):
                                        e.step()
                                        box_valid = target in box_cells - set(intersection) - set(bc) and digit in vb | (v - vr)
                                        row_valid = target in row_cells - set(intersection) - set(rc) and digit in vr | (v - vb)
                                        if box_valid or row_valid:
                                            e.emit(describe_sue_de_coq(ctx, intersection, bc, rc,
                                                                      (kind, index), box, digit, target))
    except _Stop:
        return e.finish(True)
    return e.finish()


def describe_aligned_exclusion(ctx, base, helpers):
    """Project base cells against visible near-locked helper sets.

    For q base cells a helper of k cells may have k+1 through k+q-1 digits.
    All helper cells must be visible to every base cell.  Base cells need not
    see one another; their actual peer conflicts enter through original links.
    This includes aligned pair and triple exclusion without manually excluding
    any value combination.
    """
    base = _cells(ctx, base, ctx.limits.max_subset_size)
    if len(base) < 2 or any(len(ctx.options(ctx.pos(*cell))) < 2 for cell in base):
        raise ValueError("Aligned exclusion needs at least two unsolved base cells")
    if not isinstance(helpers, (tuple, list)) or not 1 <= len(helpers) <= ctx.limits.max_petals:
        raise ValueError("Aligned exclusion needs bounded explicit helper sets")
    selected, base_digits = set(base), _options(ctx, base)
    normalized = []
    for cells in helpers:
        cells = _cells(ctx, cells, ctx.limits.max_als_size)
        digits = _options(ctx, cells)
        excess = len(digits) - len(cells)
        if (set(cells) & set(base) or not _houses(ctx, cells)
                or not 1 <= excess < len(base)
                or any(len(ctx.options(ctx.pos(*cell))) < 2 for cell in cells)
                or not all(_sees_all(ctx, cell, cells) for cell in base)
                or len(digits & base_digits) < excess + 1):
            raise ValueError("Invalid visible near-locked exclusion helper")
        selected.update(cells)
        normalized.append(cells)
    return _motif(ctx, "aligned_exclusion", selected, base,
                  ("aligned exclusion", base, tuple(normalized)))


def detect_aligned_exclusion(ctx, size=2):
    if (isinstance(size, bool) or not isinstance(size, Integral)
            or not 2 <= size <= ctx.limits.max_subset_size):
        raise ValueError("Aligned base size must be within max_subset_size")
    if size > ctx.n ** 2:
        return ctx.scan().finish(())
    e = _Enumeration(ctx)
    try:
        candidates = _board(ctx, e)
        helpers = _catalogue(ctx, e, candidates, tuple(range(1, min(size, ctx.n))))
        eligible = tuple(cell for cell, options in candidates.items() if len(options) >= 2)
        for base in combinations(eligible, size):
            e.step()
            base_digits = frozenset().union(*(candidates[cell] for cell in base))
            selected = []
            for helper in helpers:
                e.step()
                if (set(base) & set(helper.cells)
                        or len(helper.digits & base_digits) < len(helper.digits) - len(helper.cells) + 1
                        or not all(_sees_all(ctx, cell, helper.cells) for cell in base)):
                    continue
                if len(selected) >= ctx.limits.max_petals:
                    e.truncated = True
                    break
                selected.append(helper.cells)
            if selected:
                e.emit(describe_aligned_exclusion(ctx, base, selected))
    except _Stop:
        return e.finish(True)
    return e.finish()


__all__ = ["AlmostLockedSet", "make_als", "describe_als_xz", "describe_doubly_linked_als",
           "detect_als_xz_extended", "describe_als_chain", "detect_als_chains", "detect_als_xy_wing",
           "describe_death_blossom", "detect_death_blossom", "describe_bent_subset",
           "detect_bent_subsets", "detect_wxyz_wings", "describe_sue_de_coq", "detect_sue_de_coq",
           "describe_aligned_exclusion", "detect_aligned_exclusion"]
