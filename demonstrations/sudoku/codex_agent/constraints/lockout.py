from __future__ import annotations

from .core_utils import binary_core, nary_core, side


def lockout_line_cores(
    lines,
    min_endpoint_difference: int = 4,
    num: int = 3,
    prefix: str = "lockout",
) -> dict[str, object]:
    board_side = side(num)
    cores = {}
    for line_index, line in enumerate(lines):
        cells = tuple(line)
        if len(cells) < 2:
            continue
        left = cells[0]
        right = cells[-1]
        endpoint_key = f"{prefix}_{line_index}_endpoints"
        cores[endpoint_key] = binary_core(
            endpoint_key,
            left,
            right,
            lambda a, b, threshold=min_endpoint_difference: abs((a + 1) - (b + 1)) >= threshold,
            num=num,
        )
        for middle_index, middle in enumerate(cells[1:-1]):
            key = f"{prefix}_{line_index}_{middle_index}"
            allowed = (
                (left_value, middle_value, right_value)
                for left_value in range(board_side)
                for middle_value in range(board_side)
                for right_value in range(board_side)
                if not min(left_value, right_value) < middle_value < max(left_value, right_value)
            )
            cores[key] = nary_core(key, (left, middle, right), allowed, num=num)
    return cores

