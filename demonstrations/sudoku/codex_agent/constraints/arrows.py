from __future__ import annotations

from .core_utils import nary_core, side


def _sum_tuples(length: int, total: int, board_side: int):
    if length == 0:
        if total == 0:
            yield ()
        return
    for value in range(board_side):
        remaining = total - (value + 1)
        if remaining < 0:
            continue
        for tail in _sum_tuples(length - 1, remaining, board_side):
            yield (value, *tail)


def arrow_cores(arrows, num: int = 3, prefix: str = "arrow") -> dict[str, object]:
    board_side = side(num)
    cores = {}
    for index, arrow in enumerate(arrows):
        cells = tuple(arrow)
        if len(cells) < 2:
            continue
        allowed = (
            (circle, *line_values)
            for circle in range(board_side)
            for line_values in _sum_tuples(len(cells) - 1, circle + 1, board_side)
        )
        cores[f"{prefix}_{index}"] = nary_core(f"{prefix}_{index}", cells, allowed, num=num)
    return cores

