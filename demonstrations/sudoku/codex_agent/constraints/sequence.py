from __future__ import annotations

from itertools import product

from .core_utils import nary_core, side


def arithmetic_sequence_line_cores(lines, num: int = 3, prefix: str = "arithmetic_sequence") -> dict[str, object]:
    board_side = side(num)
    cores = {}
    for index, line in enumerate(lines):
        cells = tuple(line)
        if len(cells) <= 2:
            continue
        allowed = (
            values
            for values in product(range(board_side), repeat=len(cells))
            if len({values[pos + 1] - values[pos] for pos in range(len(values) - 1)}) == 1
            and values[1] != values[0]
        )
        cores[f"{prefix}_{index}"] = nary_core(f"{prefix}_{index}", cells, allowed, num=num)
    return cores

