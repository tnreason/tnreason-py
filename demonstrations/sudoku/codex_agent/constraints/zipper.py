from __future__ import annotations

from .core_utils import nary_core, side


def zipper_line_cores(lines, num: int = 3, prefix: str = "zipper") -> dict[str, object]:
    board_side = side(num)
    cores = {}
    for line_index, line in enumerate(lines):
        cells = tuple(line)
        center_index = len(cells) // 2
        center = cells[center_index]
        for pair_index in range(center_index):
            left = cells[pair_index]
            right = cells[-pair_index - 1]
            key = f"{prefix}_{line_index}_{pair_index}"
            allowed = (
                (left_value, center_value, right_value)
                for left_value in range(board_side)
                for center_value in range(board_side)
                for right_value in range(board_side)
                if (left_value + 1) + (right_value + 1) == center_value + 1
            )
            cores[key] = nary_core(key, (left, center, right), allowed, num=num)
    return cores

