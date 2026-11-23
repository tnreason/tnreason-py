from __future__ import annotations

from .core_utils import binary_core


def white_dot_allowed(left: int, right: int) -> bool:
    return abs(left - right) == 1


def black_dot_allowed(left: int, right: int) -> bool:
    left_digit = left + 1
    right_digit = right + 1
    return left_digit == 2 * right_digit or right_digit == 2 * left_digit


def xv_sum_allowed(total: int):
    return lambda left, right: (left + 1) + (right + 1) == total


def grey_dot_allowed(left: int, right: int) -> bool:
    return white_dot_allowed(left, right) or black_dot_allowed(left, right)


def edge_marker_cores(markers, num: int = 3, prefix: str = "edge") -> dict[str, object]:
    predicates = {
        "white_dot": white_dot_allowed,
        "black_dot": black_dot_allowed,
        "v_sum": xv_sum_allowed(5),
        "x_sum": xv_sum_allowed(10),
        "grey_dot": grey_dot_allowed,
    }
    cores = {}
    for index, marker in enumerate(markers):
        kind, left, right = marker[:3]
        if kind not in predicates:
            raise ValueError(f"Unknown edge marker kind {kind!r}")
        cores[f"{prefix}_{kind}_{index}"] = binary_core(
            f"{prefix}_{kind}_{index}",
            left,
            right,
            predicates[kind],
            num=num,
        )
    return cores

