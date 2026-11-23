from __future__ import annotations

from .core_utils import binary_core, side, unary_core


def is_odd_digit(value: int) -> bool:
    return (value + 1) % 2 == 1


def is_even_digit(value: int) -> bool:
    return (value + 1) % 2 == 0


def parity_edge_cores(edges, num: int = 3, prefix: str = "parity_edge") -> dict[str, object]:
    return {
        f"{prefix}_{index}": binary_core(
            f"{prefix}_{index}",
            left,
            right,
            lambda a, b: is_odd_digit(a) != is_odd_digit(b),
            num=num,
        )
        for index, (left, right) in enumerate(edges)
    }


def odd_even_cell_cores(
    odd_cells=(),
    even_cells=(),
    num: int = 3,
    prefix: str = "parity_cell",
) -> dict[str, object]:
    board_side = side(num)
    cores = {}
    for index, coord in enumerate(odd_cells):
        cores[f"{prefix}_odd_{index}"] = unary_core(
            f"{prefix}_odd_{index}",
            coord,
            (value for value in range(board_side) if is_odd_digit(value)),
            num=num,
        )
    for index, coord in enumerate(even_cells):
        cores[f"{prefix}_even_{index}"] = unary_core(
            f"{prefix}_even_{index}",
            coord,
            (value for value in range(board_side) if is_even_digit(value)),
            num=num,
        )
    return cores

