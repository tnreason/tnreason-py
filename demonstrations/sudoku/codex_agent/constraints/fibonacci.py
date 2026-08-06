from __future__ import annotations

from itertools import product

from .core_utils import nary_core, side


def _is_fibonacci_sequence(values: tuple[int, ...]) -> bool:
    digits = tuple(value + 1 for value in values)
    if len(digits) <= 1:
        return True
    if not all(left < right for left, right in zip(digits, digits[1:])):
        return False
    return all(digits[index] == digits[index - 1] + digits[index - 2] for index in range(2, len(digits)))


def fibonacci_line_cores(lines, num: int = 3, prefix: str = "fibonacci") -> dict[str, object]:
    board_side = side(num)
    cores = {}
    for index, line in enumerate(lines):
        cells = tuple(line)
        allowed = (
            values
            for values in product(range(board_side), repeat=len(cells))
            if _is_fibonacci_sequence(values) or _is_fibonacci_sequence(tuple(reversed(values)))
        )
        cores[f"{prefix}_{index}"] = nary_core(f"{prefix}_{index}", cells, allowed, num=num)
    return cores

