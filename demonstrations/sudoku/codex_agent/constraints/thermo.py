from __future__ import annotations

from .core_utils import binary_core


def thermo_line_cores(
    lines,
    strict: bool = True,
    num: int = 3,
    prefix: str = "thermo",
) -> dict[str, object]:
    cores = {}
    for line_index, line in enumerate(lines):
        for pair_index, (left, right) in enumerate(zip(line, line[1:])):
            key = f"{prefix}_{line_index}_{pair_index}"
            if strict:
                predicate = lambda a, b: a < b
            else:
                predicate = lambda a, b: a <= b
            cores[key] = binary_core(key, left, right, predicate, num=num)
    return cores

