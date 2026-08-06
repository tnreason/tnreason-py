from __future__ import annotations

from .arrows import _sum_tuples
from .core_utils import coord_to_rc, nary_core, rc_to_coord, side


_DIRECTION_DELTAS = {
    "lower right": (1, 1),
    "lower left": (1, -1),
    "upper right": (-1, 1),
    "upper left": (-1, -1),
}


def diagonal_cells(start_coord: str, direction: str, num: int = 3) -> tuple[str, ...]:
    row, col = coord_to_rc(start_coord)
    if direction not in _DIRECTION_DELTAS:
        raise ValueError(f"Unknown little-killer direction {direction!r}")
    row_delta, col_delta = _DIRECTION_DELTAS[direction]
    board_side = side(num)
    cells = []
    row += row_delta
    col += col_delta
    while 0 <= row < board_side and 0 <= col < board_side:
        cells.append(rc_to_coord(row, col))
        row += row_delta
        col += col_delta
    return tuple(cells)


def little_killer_cores(little_killers, num: int = 3, prefix: str = "little_killer") -> dict[str, object]:
    board_side = side(num)
    cores = {}
    for index, clue in enumerate(little_killers):
        cells, total = clue
        cells = tuple(cells)
        allowed = _sum_tuples(len(cells), int(total), board_side)
        cores[f"{prefix}_{index}"] = nary_core(f"{prefix}_{index}", cells, allowed, num=num)
    return cores

