from __future__ import annotations

from .core_utils import nary_core, side


def four_cell_equal_sum_cores(lines, num: int = 3, prefix: str = "equal_sum") -> dict[str, object]:
    board_side = side(num)
    cores = {}
    for index, line in enumerate(lines):
        cells = tuple(line)
        if len(cells) != 4:
            raise ValueError(f"Equal-sum line must contain four cells, got {cells!r}")
        allowed = (
            (a, b, c, d)
            for a in range(board_side)
            for b in range(board_side)
            for c in range(board_side)
            for d in range(board_side)
            if (a + 1) + (b + 1) == (c + 1) + (d + 1)
        )
        cores[f"{prefix}_{index}"] = nary_core(f"{prefix}_{index}", cells, allowed, num=num)
    return cores

