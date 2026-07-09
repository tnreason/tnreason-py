from __future__ import annotations

from .core_utils import binary_core


def palindrome_line_cores(lines, num: int = 3, prefix: str = "palindrome") -> dict[str, object]:
    cores = {}
    for line_index, line in enumerate(lines):
        cells = tuple(line)
        for pair_index in range(len(cells) // 2):
            key = f"{prefix}_{line_index}_{pair_index}"
            cores[key] = binary_core(
                key,
                cells[pair_index],
                cells[-pair_index - 1],
                lambda a, b: a == b,
                num=num,
            )
    return cores

