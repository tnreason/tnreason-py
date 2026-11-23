from __future__ import annotations

from .core_utils import nary_core, side


def modular_line_cores(lines, modulus: int = 3, num: int = 3, prefix: str = "modular") -> dict[str, object]:
    board_side = side(num)
    cores = {}
    for line_index, line in enumerate(lines):
        for window_index, window in enumerate(zip(line, line[1:], line[2:])):
            key = f"{prefix}_{line_index}_{window_index}"
            allowed = (
                (a, b, c)
                for a in range(board_side)
                for b in range(board_side)
                for c in range(board_side)
                if {a % modulus, b % modulus, c % modulus} == set(range(modulus))
            )
            cores[key] = nary_core(key, window, allowed, num=num)
    return cores

