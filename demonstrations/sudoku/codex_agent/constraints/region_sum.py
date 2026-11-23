from __future__ import annotations

from itertools import product

from .core_utils import cell_color, coord_to_rc, drop_closing_repeat, side, table_core


def _box(coord: str, num: int = 3) -> tuple[int, int]:
    row, col = coord_to_rc(coord)
    return row // num, col // num


def _segments(line, num: int = 3) -> tuple[tuple[str, ...], ...]:
    segments = []
    current = []
    current_box = None
    for coord in line:
        coord_box = _box(coord, num=num)
        if current and coord_box != current_box:
            segments.append(tuple(current))
            current = []
        current.append(coord)
        current_box = coord_box
    if current:
        segments.append(tuple(current))
    return tuple(segment for segment in segments if segment)


def region_sum_line_cores(lines, num: int = 3, prefix: str = "region_sum") -> dict[str, object]:
    board_side = side(num)
    cores = {}
    for line_index, line in enumerate(lines):
        segments = _segments(drop_closing_repeat(line), num=num)
        if len(segments) < 2:
            continue
        max_sum = max(len(segment) for segment in segments) * board_side
        aux_color = f"{prefix}_{line_index}_total"
        for segment_index, segment in enumerate(segments):
            colors = (*[cell_color(coord, num=num) for coord in segment], aux_color)
            shape = (*[board_side for _ in segment], max_sum + 1)
            allowed = (
                (*values, sum(value + 1 for value in values))
                for values in product(range(board_side), repeat=len(segment))
            )
            key = f"{prefix}_{line_index}_{segment_index}"
            cores[key] = table_core(key, colors, allowed, shape=shape, num=num)
    return cores
