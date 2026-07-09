from __future__ import annotations

from .core_utils import binary_core, rc_to_coord, side


def anti_knight_cores(enabled: bool = True, num: int = 3, prefix: str = "anti_knight") -> dict[str, object]:
    if not enabled:
        return {}
    board_side = side(num)
    offsets = ((1, 2), (2, 1), (-1, 2), (2, -1))
    cores = {}
    index = 0
    for row in range(board_side):
        for col in range(board_side):
            for row_delta, col_delta in offsets:
                other_row = row + row_delta
                other_col = col + col_delta
                if 0 <= other_row < board_side and 0 <= other_col < board_side:
                    key = f"{prefix}_{index}"
                    cores[key] = binary_core(
                        key,
                        rc_to_coord(row, col),
                        rc_to_coord(other_row, other_col),
                        lambda a, b: a != b,
                        num=num,
                    )
                    index += 1
    return cores

