from .naked_pairs import _iterate_units, _candidate_digits_for_cell, _append_pair_cluster
from itertools import combinations

def detect_hidden_pairs(propagator, num=3):
    """
    Hidden-pair detector (digit-centric):
    if two digits in a unit can only appear in the same two cells, the cells form a locked pair.
    """
    out = []
    seen = set()

    for unit_cells, unit_key_fn in _iterate_units(num):
        digit_to_cells = {}
        for n in range(num ** 2):
            digit_to_cells[n] = {
                cell for cell in unit_cells if n in _candidate_digits_for_cell(propagator, cell, num)
            }
        for n1, n2 in combinations(range(num ** 2), 2):
            if len(digit_to_cells[n1]) == 2 and digit_to_cells[n1] == digit_to_cells[n2]:
                ca, cb = list(digit_to_cells[n1])
                _append_pair_cluster(
                    out, seen, unit_key_fn(n1), unit_key_fn(n2), ca, cb, n1, n2
                )

    if out:
        print(f"Hidden pairs detected: {len(out)}")
    return out