from __future__ import annotations

from itertools import combinations, permutations, product

from .core_utils import binary_core, nary_core, side, variant_unit_cores


def _coerce_total(total):
    if total in (None, ""):
        return None
    return int(total)


def no_repeat_cage_cores(cages, num: int = 3, prefix: str = "norepeat_cage") -> dict[str, object]:
    board_side = side(num)
    unit_cages = []
    cores = {}
    pair_index = 0
    for cage_index, cells in enumerate(cages):
        cells = tuple(cells)
        if len(cells) == board_side:
            unit_cages.append(cells)
            continue
        for left, right in combinations(cells, 2):
            key = f"{prefix}_{cage_index}_{pair_index}"
            cores[key] = binary_core(key, left, right, lambda a, b: a != b, num=num)
            pair_index += 1
    cores.update(variant_unit_cores(unit_cages, num=num, prefix=prefix))
    return cores


def killer_cage_cores(cages, num: int = 3, prefix: str = "killer") -> dict[str, object]:
    board_side = side(num)
    cores = {}
    no_repeat_only = []
    for index, cage in enumerate(cages):
        cells, total = cage[:2]
        no_repeat = True if len(cage) < 3 else bool(cage[2])
        cells = tuple(cells)
        total = _coerce_total(total)
        if total is None:
            if no_repeat:
                no_repeat_only.append(cells)
            continue
        if no_repeat:
            candidates = (
                values
                for values in permutations(range(board_side), len(cells))
                if sum(value + 1 for value in values) == total
            )
        else:
            candidates = (
                values
                for values in product(range(board_side), repeat=len(cells))
                if sum(value + 1 for value in values) == total
            )
        cores[f"{prefix}_sum_{index}"] = nary_core(
            f"{prefix}_sum_{index}",
            cells,
            candidates,
            num=num,
        )
    cores.update(no_repeat_cage_cores(no_repeat_only, num=num, prefix=f"{prefix}_norepeat"))
    return cores

