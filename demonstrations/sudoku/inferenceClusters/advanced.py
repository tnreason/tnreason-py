"""Bounded, opt-in wings, simple coloring, XY chains and disjoint ALS-XZ.

Detectors select original position/unit-to-atom tensors.  Their conclusions are
*not* applied here: ``contract_cluster`` computes support under the context's
domain snapshot.  No uniqueness assumption or solved grid is used.

Simple coloring enumerates conjugate paths up to ``max_chain_length`` cells,
including color traps and color wraps.  It does not claim full-component,
multi-color, general AIC/nice-loop, grouped-link or forcing-net coverage.
ALS-XZ covers disjoint ALS pairs and ordinary XZ eliminations; overlap, special
doubly-linked eliminations and ALS chains remain unsupported.  Strong links
are read from the unit variable's own current domain, so GAC-closed input
domains normally expose more patterns than unpropagated position domains.

Definitions: https://hodoku.sourceforge.net/en/tech_wings.php,
https://hodoku.sourceforge.net/en/tech_col.php,
https://hodoku.sourceforge.net/en/tech_chains.php,
https://hodoku.sourceforge.net/en/tech_als.php.
"""

from itertools import combinations


class _StopScan(Exception):
    pass


class _Enumeration:
    """Share the caller's finite work/pattern budget across nested searches."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.scan = ctx.scan()
        self.out = []
        self.seen = set()

    def step(self):
        if not self.scan.step():
            raise _StopScan

    def emit(self, family, cells, edges, targets, evidence=(), units=()):
        categories = {self.ctx.pos(*cell) for cell in cells}
        categories.update(units)
        for first, second, digit in edges:
            categories.update(_shared_units(self.ctx, first, second, digit))
        outputs = tuple(sorted(self.ctx.pos(*cell) for cell in targets))
        signature = tuple(sorted(categories)), outputs
        if signature in self.seen:
            return
        if len(self.out) >= self.ctx.limits.max_patterns:
            raise _StopScan
        self.seen.add(signature)
        self.out.append(self.ctx.describe(family, sorted(categories), outputs=outputs,
                                          evidence=evidence))

    def finish(self, truncated=False):
        return self.scan.finish(self.out, truncated=truncated)


def _board(ctx, enumeration):
    candidates = {}
    for row in range(ctx.n):
        for col in range(ctx.n):
            enumeration.step()
            candidates[row, col] = ctx.options(ctx.pos(row, col))
    return candidates


def _sees(ctx, first, second):
    return first != second and (
        first[0] == second[0] or first[1] == second[1]
        or (first[0] // ctx.num, first[1] // ctx.num)
        == (second[0] // ctx.num, second[1] // ctx.num))


def _shared_units(ctx, first, second, digit):
    if first == second:
        return ()
    units = []
    if first[0] == second[0]:
        units.append(ctx.unit("row", first[0], digit))
    if first[1] == second[1]:
        units.append(ctx.unit("col", first[1], digit))
    if (first[0] // ctx.num, first[1] // ctx.num) == (
            second[0] // ctx.num, second[1] // ctx.num):
        square = first[0] // ctx.num * ctx.num + first[1] // ctx.num
        units.append(ctx.unit("square", square, digit))
    return units


def _unit_cells(ctx, kind, index):
    if kind == "row":
        return tuple((index, col) for col in range(ctx.n))
    if kind == "col":
        return tuple((row, index) for row in range(ctx.n))
    row, col = index // ctx.num * ctx.num, index % ctx.num * ctx.num
    return tuple((row + offset // ctx.num, col + offset % ctx.num)
                 for offset in range(ctx.n))


def _targets(ctx, enumeration, candidates, digit, sources, excluded):
    for cell, options in candidates.items():
        enumeration.step()
        if cell not in excluded and digit in options and all(
                _sees(ctx, cell, source) for source in sources):
            yield cell


def _detect_wings(ctx, xyz):
    e = _Enumeration(ctx)
    try:
        candidates = _board(ctx, e)
        bivalue = {cell: opts for cell, opts in candidates.items() if len(opts) == 2}
        for pivot, digits in candidates.items():
            e.step()
            if len(digits) != (3 if xyz else 2):
                continue
            pincers = []
            for cell, opts in bivalue.items():
                e.step()
                if _sees(ctx, pivot, cell) and (
                        opts < digits if xyz else len(opts & digits) == 1):
                    pincers.append(cell)
            for first, second in combinations(pincers, 2):
                e.step()
                a, b = bivalue[first], bivalue[second]
                common = a & b
                if len(common) != 1 or len(a | b) != 3:
                    continue
                z = next(iter(common))
                if not xyz and (z in digits or (a | b) - {z} != digits):
                    continue
                if xyz and a | b != digits:
                    continue
                x, y = next(iter(a - {z})), next(iter(b - {z}))
                sources = (first, second, pivot) if xyz else (first, second)
                for target in _targets(ctx, e, candidates, z, sources,
                                       {pivot, first, second}):
                    edges = [(pivot, first, x), (pivot, second, y),
                             (first, target, z), (second, target, z)]
                    if xyz:
                        edges.append((pivot, target, z))
                    e.emit("xyz_wing" if xyz else "xy_wing",
                           (pivot, first, second, target), edges, (target,),
                           evidence=(f"pivot={pivot}; pincers={first},{second}; digit={z}",))
    except _StopScan:
        return e.finish(truncated=True)
    return e.finish()


def detect_xy_wings(ctx):
    """XY pivot, XZ/YZ bivalue pincers; target sees both pincers."""
    return _detect_wings(ctx, xyz=False)


def detect_xyz_wings(ctx):
    """XYZ pivot, XZ/YZ pincers; target sees pivot and both pincers."""
    return _detect_wings(ctx, xyz=True)


def detect_xy_chains(ctx):
    """Enumerate simple bivalue XY chains with at most the configured cells.

    Consecutive cells see each other in their linking digit.  Both endpoint
    cells contain the same unlinked digit Z; a target sees both endpoints.
    XY-Wings and remote-pair paths are parameterized special cases.
    """
    e = _Enumeration(ctx)
    try:
        candidates = _board(ctx, e)
        bivalue = {cell: opts for cell, opts in candidates.items() if len(opts) == 2}
        neighbors = {cell: [] for cell in bivalue}
        for first, second in combinations(bivalue, 2):
            e.step()
            if _sees(ctx, first, second):
                neighbors[first].append(second)
                neighbors[second].append(first)

        def walk(path, edges, z, exit_digit):
            e.step()
            if len(path) >= ctx.limits.max_chain_length:
                return
            for cell in neighbors[path[-1]]:
                e.step()
                if cell in path or exit_digit not in bivalue[cell]:
                    continue
                next_digit = next(iter(bivalue[cell] - {exit_digit}))
                new_path = (*path, cell)
                new_edges = (*edges, (path[-1], cell, exit_digit))
                if len(new_path) >= 3 and next_digit == z and new_path[0] < cell:
                    for target in _targets(ctx, e, candidates, z,
                                           (new_path[0], cell), set(new_path)):
                        e.emit("xy_chain", (*new_path, target),
                               (*new_edges, (new_path[0], target, z),
                                (cell, target, z)), (target,),
                               evidence=(f"cells={new_path}; endpoint digit={z}",))
                walk(new_path, new_edges, z, next_digit)

        for cell, digits in bivalue.items():
            for z in sorted(digits):
                e.step()
                walk((cell,), (), z, next(iter(digits - {z})))
    except _StopScan:
        return e.finish(truncated=True)
    return e.finish()


def detect_simple_coloring(ctx):
    """Bounded conjugate paths: opposite-color traps and same-color wraps.

    ``max_chain_length`` bounds the number of candidate cells in a path.  A
    unit is a strong edge only if its own current options contain two cells.
    Longer paths are outside configured coverage, even when scan completes.
    """
    e = _Enumeration(ctx)
    try:
        candidates = _board(ctx, e)
        for digit in range(ctx.n):
            graph = {}
            for kind in ("row", "col", "square"):
                for index in range(ctx.n):
                    e.step()
                    unit = ctx.unit(kind, index, digit)
                    options = sorted(ctx.options(unit))
                    if len(options) != 2:
                        continue
                    cells = _unit_cells(ctx, kind, index)
                    first, second = (cells[option] for option in options)
                    if any(digit not in candidates[cell] for cell in (first, second)):
                        continue
                    graph.setdefault(first, []).append((second, unit))
                    graph.setdefault(second, []).append((first, unit))

            def walk(path, units):
                e.step()
                if len(path) >= 2 and path[0] < path[-1]:
                    if len(path) % 2 == 0:
                        for target in _targets(ctx, e, candidates, digit,
                                               (path[0], path[-1]), set(path)):
                            e.emit("simple_coloring", (*path, target),
                                   ((path[0], target, digit),
                                    (path[-1], target, digit)), (target,),
                                   units=units,
                                   evidence=(f"color trap; digit={digit}; path={path}",))
                    elif _sees(ctx, path[0], path[-1]):
                        e.emit("simple_coloring", path,
                               ((path[0], path[-1], digit),), path[::2],
                               units=units,
                               evidence=(f"color wrap; digit={digit}; path={path}",))
                if len(path) >= ctx.limits.max_chain_length:
                    return
                for cell, unit in graph[path[-1]]:
                    e.step()
                    if cell not in path:
                        walk((*path, cell), (*units, unit))

            for start in sorted(graph):
                walk((start,), ())
    except _StopScan:
        return e.finish(truncated=True)
    return e.finish()


def detect_als_xz(ctx):
    """Disjoint ALS pairs, an RCC X, and common elimination digit Z != X.

    Each ALS is k unsolved cells in one unit with k+1 candidate digits,
    1 <= k <= ``max_als_size``.  Every X occurrence in one ALS sees every X
    occurrence in the other.  The target sees every Z occurrence in both.
    An RCC can be selected even if another RCC exists; only the ordinary XZ
    elimination is described, without claiming doubly-linked-specific rules.
    """
    e = _Enumeration(ctx)
    try:
        candidates = _board(ctx, e)
        als = {}
        for kind in ("row", "col", "square"):
            for index in range(ctx.n):
                e.step()
                cells = [cell for cell in _unit_cells(ctx, kind, index)
                         if len(candidates[cell]) >= 2]
                for size in range(1, min(ctx.limits.max_als_size, len(cells)) + 1):
                    for chosen in combinations(cells, size):
                        e.step()
                        digits = frozenset().union(*(candidates[cell] for cell in chosen))
                        if len(digits) == size + 1:
                            key = tuple(sorted(chosen))
                            als.setdefault(key, (digits, kind, index))
        for first, second in combinations(als, 2):
            e.step()
            if set(first) & set(second):
                continue
            a, akind, aindex = als[first]
            b, bkind, bindex = als[second]
            common = a & b
            if len(common) < 2:
                continue
            for x in sorted(common):
                e.step()
                ax = [cell for cell in first if x in candidates[cell]]
                bx = [cell for cell in second if x in candidates[cell]]
                rcc_edges = []
                restricted = True
                for ca in ax:
                    for cb in bx:
                        e.step()
                        if not _sees(ctx, ca, cb):
                            restricted = False
                        rcc_edges.append((ca, cb, x))
                if not restricted:
                    continue
                # Within each ALS every candidate digit is constrained to occur
                # at most once by that ALS's original house variable.
                als_units = [ctx.unit(akind, aindex, digit) for digit in a]
                als_units += [ctx.unit(bkind, bindex, digit) for digit in b]
                for z in sorted(common - {x}):
                    sources = tuple(cell for cell in (*first, *second)
                                    if z in candidates[cell])
                    for target in _targets(ctx, e, candidates, z, sources,
                                           set(first) | set(second)):
                        e.emit("als_xz", (*first, *second, target),
                               (*rcc_edges, *((cell, target, z) for cell in sources)),
                               (target,), units=als_units,
                               evidence=(f"ALS A={first}; B={second}; RCC={x}; Z={z}",))
    except _StopScan:
        return e.finish(truncated=True)
    return e.finish()


__all__ = ["detect_xy_wings", "detect_xyz_wings", "detect_xy_chains",
           "detect_simple_coloring", "detect_als_xz"]
