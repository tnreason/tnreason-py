from __future__ import annotations

from .core_utils import nary_core, side


def entropic_line_cores(lines, num: int = 3, prefix: str = "entropy") -> dict[str, object]:
    board_side = side(num)
    group_size = num
    cores = {}
    for line_index, line in enumerate(lines):
        for window_index, window in enumerate(zip(line, line[1:], line[2:])):
            key = f"{prefix}_{line_index}_{window_index}"
            allowed = (
                (a, b, c)
                for a in range(board_side)
                for b in range(board_side)
                for c in range(board_side)
                if {a // group_size, b // group_size, c // group_size} == set(range(group_size))
            )
            cores[key] = nary_core(key, window, allowed, num=num)
    return cores

