from __future__ import annotations

from collections import Counter
from itertools import product

from .core_utils import nary_core, side


def quadruple_cores(quadruples, num: int = 3, prefix: str = "quadruple") -> dict[str, object]:
    board_side = side(num)
    cores = {}
    for index, quadruple in enumerate(quadruples):
        cells, digits = quadruple
        cells = tuple(cells)
        required = Counter(int(digit) - 1 for digit in digits)
        allowed = (
            values
            for values in product(range(board_side), repeat=len(cells))
            if all(Counter(values)[digit] >= count for digit, count in required.items())
        )
        cores[f"{prefix}_{index}"] = nary_core(f"{prefix}_{index}", cells, allowed, num=num)
    return cores

