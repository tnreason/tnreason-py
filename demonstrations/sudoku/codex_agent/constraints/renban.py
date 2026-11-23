from __future__ import annotations

from itertools import permutations

from .core_utils import nary_core, side, unique_coords


def _renban_tuples(length: int, board_side: int):
    for start in range(board_side - length + 1):
        values = tuple(range(start, start + length))
        yield from permutations(values)


def renban_line_cores(lines, num: int = 3, prefix: str = "renban") -> dict[str, object]:
    board_side = side(num)
    cores = {}
    for index, line in enumerate(lines):
        cells = unique_coords(tuple(line))
        cores[f"{prefix}_{index}"] = nary_core(
            f"{prefix}_{index}",
            cells,
            _renban_tuples(len(cells), board_side),
            num=num,
        )
    return cores
