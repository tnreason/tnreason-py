from itertools import combinations


def _append_pair_cluster(out, seen, unit_key_1, unit_key_2, cell_a, cell_b, d1, d2):
    ca, cb = sorted([cell_a, cell_b])
    n1, n2 = sorted([d1, d2])
    uk1, uk2 = sorted([unit_key_1, unit_key_2])
    signature = (uk1, uk2, ca, cb, n1, n2)
    if signature in seen:
        return
    seen.add(signature)

    r1a, r2a, c1a, c2a = ca
    r1b, r2b, c1b, c2b = cb
    out.append([
        f"pos_{r1a}_{r2a}_{c1a}_{c2a}",
        f"pos_{r1b}_{r2b}_{c1b}_{c2b}",
        uk1,
        uk2,
        f"a_{r1a}_{r2a}_{c1a}_{c2a}_{n1}",
        f"a_{r1a}_{r2a}_{c1a}_{c2a}_{n2}",
        f"a_{r1b}_{r2b}_{c1b}_{c2b}_{n1}",
        f"a_{r1b}_{r2b}_{c1b}_{c2b}_{n2}",
    ])


def _iterate_units(num):
    # rows
    for r1 in range(num):
        for r2 in range(num):
            unit_cells = [(r1, r2, c1, c2) for c1 in range(num) for c2 in range(num)]
            unit_key_fn = lambda n, r1=r1, r2=r2: f"row_{n}_{r1}_{r2}"
            yield unit_cells, unit_key_fn
    # cols
    for c1 in range(num):
        for c2 in range(num):
            unit_cells = [(r1, r2, c1, c2) for r1 in range(num) for r2 in range(num)]
            unit_key_fn = lambda n, c1=c1, c2=c2: f"col_{n}_{c1}_{c2}"
            yield unit_cells, unit_key_fn
    # squares
    for sr1 in range(num):
        for sc1 in range(num):
            unit_cells = [(sr1, r2, sc1, c2) for r2 in range(num) for c2 in range(num)]
            unit_key_fn = lambda n, sr1=sr1, sc1=sc1: f"square_{n}_{sr1}_{sc1}"
            yield unit_cells, unit_key_fn


def _candidate_digits_for_cell(propagator, cell, num):
    r1, r2, c1, c2 = cell
    pos_key = f"pos_{r1}_{r2}_{c1}_{c2}"
    return {
        n for n in range(num ** 2)
        if propagator.meanParamDict[pos_key][{pos_key: n}] > 0.5
    }


def detect_naked_pairs(propagator, num=3):
    """
    Naked-pair detector (cell-centric):
    if two cells in a unit have exactly the same two candidates, those digits are excluded from other cells.

    Returns inference clusters linking:
    - both pair cells via pos_* keys
    - both unit-digit constraints (row/col/square)
    - the 4 atom keys (2 cells x 2 digits)
    """
    out = []
    seen = set()

    for unit_cells, unit_key_fn in _iterate_units(num):
        pair_cells = [
            (cell, _candidate_digits_for_cell(propagator, cell, num))
            for cell in unit_cells
        ]
        pair_cells = [(cell, cands) for cell, cands in pair_cells if len(cands) == 2]
        for (cell_a, cands_a), (cell_b, cands_b) in combinations(pair_cells, 2):
            if cands_a == cands_b:
                d1, d2 = sorted(list(cands_a))
                _append_pair_cluster(
                    out, seen, unit_key_fn(d1), unit_key_fn(d2), cell_a, cell_b, d1, d2
                )

    if out:
        print(f"Naked pairs detected: {len(out)}")
    return out
