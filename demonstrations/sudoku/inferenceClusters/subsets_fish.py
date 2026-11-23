"""Bounded selectors for single, intersection, subset and fish clusters.

These functions select *original* categorical/Boolean links.  They never apply
the textbook elimination themselves: :func:`contract_cluster` projects the
selected links with the context's immutable domain snapshot.  In particular,
the pattern tests are useful selection heuristics even when original link
weights have been changed; the actual deduction always uses realized support.
"""

from itertools import combinations


def _sizes(requested, maximum, n):
    if requested is not None and not isinstance(requested, (tuple, list, range)):
        raise TypeError("Pattern sizes must be a finite tuple, list, or range")
    if requested is not None and len(requested) > maximum:
        raise ValueError("Too many requested pattern sizes")
    values = range(2, min(maximum, n - 1) + 1) if requested is None else requested
    values = tuple(sorted(set(values)))
    if any(isinstance(size, bool) or not isinstance(size, int) or size < 2 for size in values):
        raise ValueError("Pattern sizes must be integers >= 2")
    if any(size > maximum for size in values):
        raise ValueError("Requested size exceeds the configured pattern bound")
    return tuple(size for size in values if size < n)


def _cells(ctx, kind, index):
    if kind == "row":
        return tuple((index, col) for col in range(ctx.n))
    if kind == "col":
        return tuple((row, index) for row in range(ctx.n))
    return tuple((index // ctx.num * ctx.num + row,
                  index % ctx.num * ctx.num + col)
                 for row in range(ctx.num) for col in range(ctx.num))


def _allowed_atoms(ctx, category):
    return frozenset(ctx.categories[category][value]
                     for value in ctx.options(category))


def _append(ctx, out, seen, family, categories, *, atoms=None, outputs=None, evidence=()):
    signature = (family, tuple(sorted(categories)),
                 None if atoms is None else tuple(sorted(atoms)))
    if signature in seen:
        return False
    seen.add(signature)
    out.append(ctx.describe(family, categories, atoms=atoms,
                            outputs=outputs, evidence=evidence))
    return len(out) >= ctx.limits.max_patterns


def detect_singles(ctx):
    """Select fixed position (naked) and fixed unit-digit (hidden) categories.

    All their Boolean atoms are retained.  This is exactly the projection
    already performed by ordinary GAC on the original binary links.
    """
    scan, out, seen = ctx.scan(), [], set()
    if ctx.limits.max_patterns == 0:
        return scan.finish(out, truncated=True)
    for category, atoms in ctx.categories.items():
        if not scan.step():
            return scan.finish(out, truncated=True)
        if len(ctx.options(category)) != 1:
            continue
        family = "naked_single" if category.startswith("pos_") else "hidden_single"
        if _append(ctx, out, seen, family, (category,), atoms=atoms,
                   outputs=atoms, evidence=(category,)):
            return scan.finish(out, truncated=True)
    return scan.finish(out)


def detect_locked(ctx):
    """Select pointing and claiming in rows and columns, both directions.

    The cluster consists of the two unit-digit categories and every atom in
    their intersection.  Its target category is the retained output.
    """
    scan, out, seen = ctx.scan(), [], set()
    if ctx.limits.max_patterns == 0:
        return scan.finish(out, truncated=True)
    for digit in range(ctx.n):
        for kind in ("row", "col"):
            for index in range(ctx.n):
                line = ctx.unit(kind, index, digit)
                for square in range(ctx.n):
                    if not scan.step():
                        return scan.finish(out, truncated=True)
                    box = ctx.unit("square", square, digit)
                    intersection = set(ctx.categories[line]) & set(ctx.categories[box])
                    if not intersection:
                        continue
                    for source, target in ((line, box), (box, line)):
                        if not scan.step():
                            return scan.finish(out, truncated=True)
                        allowed = _allowed_atoms(ctx, source)
                        if (len(allowed) < 2 or not allowed <= intersection
                                or not _allowed_atoms(ctx, target) - intersection):
                            continue
                        if _append(ctx, out, seen, "locked_candidates", (source, target),
                                   atoms=intersection, outputs=(target,),
                                   evidence=("claiming" if source == line else "pointing",
                                             source, target)):
                            return scan.finish(out, truncated=True)
    return scan.finish(out)


def detect_subsets(ctx, hidden=False, sizes=None):
    """Select naked/hidden Hall subsets of configurable sizes (default 2..4).

    For k cells and k digits the selected links are their k-by-k incidence
    rectangle, between position categories and the corresponding unit-digit
    categories.  This includes pairs, triples and quads with unequal candidate
    sets; it does not require the same candidate set in each cell or digit.
    """
    sizes = _sizes(sizes, ctx.limits.max_subset_size, ctx.n)
    scan, out, seen = ctx.scan(), [], set()
    if ctx.limits.max_patterns == 0:
        return scan.finish(out, truncated=True)
    for kind in ("row", "col", "square"):
        for index in range(ctx.n):
            if not scan.step():
                return scan.finish(out, truncated=True)
            cells = _cells(ctx, kind, index)
            positions = tuple(ctx.pos(*cell) for cell in cells)
            units = tuple(ctx.unit(kind, index, digit) for digit in range(ctx.n))
            options = tuple(ctx.options(category) for category in (units if hidden else positions))
            for size in sizes:
                eligible = tuple(i for i, allowed in enumerate(options) if 1 < len(allowed) <= size)
                for chosen in combinations(eligible, size):
                    if not scan.step():
                        return scan.finish(out, truncated=True)
                    union = frozenset().union(*(options[i] for i in chosen))
                    if len(union) != size:
                        continue
                    cell_indices, digits = (tuple(sorted(union)), chosen) if hidden else (chosen, tuple(sorted(union)))
                    selected_positions = tuple(positions[i] for i in cell_indices)
                    selected_units = tuple(units[digit] for digit in digits)
                    # Only describe patterns with a still-present textbook target.
                    if hidden:
                        active = any(ctx.options(position) - set(digits) for position in selected_positions)
                    else:
                        active = any(ctx.options(unit) - set(cell_indices) for unit in selected_units)
                    if not active:
                        continue
                    family = "hidden_subset" if hidden else "naked_subset"
                    if _append(ctx, out, seen, family, selected_positions + selected_units,
                               evidence=(kind, index, size, cell_indices, tuple(digits))):
                        return scan.finish(out, truncated=True)
    return scan.finish(out)


def detect_fish(ctx, sizes=None, kind="basic"):
    """Select basic, finned or sashimi row/column fish, sizes 2..4 by default.

    Basic fish retain the cover categories of a k-base/k-cover incidence
    rectangle.  Finned/sashimi fish add the one box containing *all* candidates
    outside the chosen cover set.  Targets are cover cells outside the bases
    in that box.  A finned fish has at least two body candidates in every base;
    a sashimi fish has a base with exactly one.  Every base must have at least
    one body candidate and at least two candidates in total.  Endo-fins and
    non-row/column (Franken/mutant) fish are intentionally outside this selector.
    """
    if kind not in {"basic", "finned", "sashimi"}:
        raise ValueError("Fish kind must be 'basic', 'finned', or 'sashimi'")
    sizes = _sizes(sizes, ctx.limits.max_fish_size, ctx.n)
    scan, out, seen = ctx.scan(), [], set()
    if ctx.limits.max_patterns == 0:
        return scan.finish(out, truncated=True)
    for digit in range(ctx.n):
        for base_kind, cover_kind in (("row", "col"), ("col", "row")):
            bases = tuple(ctx.unit(base_kind, index, digit) for index in range(ctx.n))
            options = tuple(ctx.options(base) for base in bases)
            for size in sizes:
                eligible = tuple(index for index, allowed in enumerate(options)
                                 if 1 < len(allowed) <= (size if kind == "basic" else size + ctx.num))
                for chosen in combinations(eligible, size):
                    if not scan.step():
                        return scan.finish(out, truncated=True)
                    union = frozenset().union(*(options[index] for index in chosen))
                    if kind == "basic":
                        if len(union) != size:
                            continue
                        covers = tuple(ctx.unit(cover_kind, index, digit) for index in sorted(union))
                        if not any(ctx.options(cover) - set(chosen) for cover in covers):
                            continue
                        if _append(ctx, out, seen, "basic_fish", tuple(bases[index] for index in chosen) + covers,
                                   outputs=covers,
                                   evidence=(base_kind, digit, size, chosen, tuple(sorted(union)))):
                            return scan.finish(out, truncated=True)
                        continue
                    if not size < len(union) <= size + ctx.num:
                        continue
                    for cover_indices in combinations(sorted(union), size):
                        if not scan.step():
                            return scan.finish(out, truncated=True)
                        cover_set = frozenset(cover_indices)
                        body_counts = tuple(len(options[index] & cover_set) for index in chosen)
                        if not all(body_counts) or ((min(body_counts) == 1) != (kind == "sashimi")):
                            continue
                        fin_cells = tuple((base, cover) if base_kind == "row" else (cover, base)
                                          for base in chosen for cover in options[base] - cover_set)
                        boxes = {row // ctx.num * ctx.num + col // ctx.num for row, col in fin_cells}
                        if len(boxes) != 1:
                            continue
                        square = next(iter(boxes))
                        targets = tuple(ctx.unit(cover_kind, cover, digit) for cover in cover_indices
                                        if any((row if base_kind == "row" else col) not in chosen
                                               and (col if base_kind == "row" else row) == cover
                                               and (row if base_kind == "row" else col)
                                               in ctx.options(ctx.unit(cover_kind, cover, digit))
                                               for row, col in _cells(ctx, "square", square)))
                        if not targets:
                            continue
                        categories = (tuple(bases[index] for index in chosen)
                                      + tuple(ctx.unit(cover_kind, index, digit) for index in cover_indices)
                                      + (ctx.unit("square", square, digit),))
                        if _append(ctx, out, seen, kind + "_fish", categories, outputs=targets,
                                   evidence=(base_kind, digit, size, chosen, cover_indices, fin_cells)):
                            return scan.finish(out, truncated=True)
    return scan.finish(out)
