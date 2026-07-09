from __future__ import annotations

from .core_utils import binary_core


def whisper_line_cores(
    lines,
    min_difference: int = 5,
    num: int = 3,
    prefix: str = "whisper",
) -> dict[str, object]:
    cores = {}
    for line_index, line in enumerate(lines):
        for pair_index, (left, right) in enumerate(zip(line, line[1:])):
            key = f"{prefix}_{line_index}_{pair_index}"
            cores[key] = binary_core(
                key,
                left,
                right,
                lambda a, b, threshold=min_difference: abs((a + 1) - (b + 1)) >= threshold,
                num=num,
            )
    return cores

