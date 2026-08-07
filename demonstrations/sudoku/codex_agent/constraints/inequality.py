from __future__ import annotations

from .core_utils import binary_core


def _predicate_for_direction(direction: str):
    if direction in (">", "v"):
        return lambda first, second: first > second
    if direction in ("<", "^"):
        return lambda first, second: first < second
    raise ValueError(f"Unknown inequality direction {direction!r}")


def inequality_cores(inequalities, num: int = 3, prefix: str = "inequality") -> dict[str, object]:
    cores = {}
    for index, inequality in enumerate(inequalities):
        direction, first, second = inequality
        key = f"{prefix}_{index}"
        cores[key] = binary_core(key, first, second, _predicate_for_direction(direction), num=num)
    return cores

