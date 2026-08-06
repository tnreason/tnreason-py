from __future__ import annotations

from experiments.constraint_networks.sudoku_tests.standard_constraints import (
    get_sudoku_constraint_network,
)

from .anti_knight import anti_knight_cores
from .arrows import arrow_cores
from .cages import killer_cage_cores, no_repeat_cage_cores
from .core_utils import evidence_cores, initial_board_to_evidence, merge_core_dicts
from .edge_markers import edge_marker_cores
from .entropy import entropic_line_cores
from .equal_sum import four_cell_equal_sum_cores
from .fibonacci import fibonacci_line_cores
from .inequality import inequality_cores
from .little_killer import little_killer_cores
from .lockout import lockout_line_cores
from .modular import modular_line_cores
from .palindrome import palindrome_line_cores
from .parity import odd_even_cell_cores, parity_edge_cores
from .quadruple import quadruple_cores
from .region_sum import region_sum_line_cores
from .renban import renban_line_cores
from .sequence import arithmetic_sequence_line_cores
from .thermo import thermo_line_cores
from .whispers import whisper_line_cores
from .zipper import zipper_line_cores


def build_variant_constraint_network(
    *,
    num: int = 3,
    start_assignment=None,
    include_initial_board: bool = True,
    initial_board: str = "",
    killer_cages=(),
    no_repeat_cages=(),
    arrows=(),
    edge_markers=(),
    strict_thermos=(),
    slow_thermos=(),
    renban_lines=(),
    whisper_lines=(),
    dutch_whisper_lines=(),
    entropic_lines=(),
    modular_lines=(),
    equal_sum_lines=(),
    zipper_lines=(),
    quadruples=(),
    palindrome_lines=(),
    anti_knight: bool = False,
    little_killers=(),
    fibonacci_lines=(),
    arithmetic_sequence_lines=(),
    lockout_lines=(),
    region_sum_lines=(),
    inequalities=(),
    parity_edges=(),
    odd_cells=(),
    even_cells=(),
) -> dict[str, object]:
    start_assignment = dict(start_assignment or {})
    constraint_cores = merge_core_dicts(
        get_sudoku_constraint_network(num=num),
        killer_cage_cores(killer_cages, num=num),
        no_repeat_cage_cores(no_repeat_cages, num=num),
        arrow_cores(arrows, num=num),
        edge_marker_cores(edge_markers, num=num),
        thermo_line_cores(strict_thermos, strict=True, num=num, prefix="strict_thermo"),
        thermo_line_cores(slow_thermos, strict=False, num=num, prefix="slow_thermo"),
        renban_line_cores(renban_lines, num=num),
        whisper_line_cores(whisper_lines, min_difference=5, num=num, prefix="german_whisper"),
        whisper_line_cores(dutch_whisper_lines, min_difference=4, num=num, prefix="dutch_whisper"),
        entropic_line_cores(entropic_lines, num=num),
        modular_line_cores(modular_lines, modulus=num, num=num),
        four_cell_equal_sum_cores(equal_sum_lines, num=num),
        zipper_line_cores(zipper_lines, num=num),
        quadruple_cores(quadruples, num=num),
        palindrome_line_cores(palindrome_lines, num=num),
        anti_knight_cores(anti_knight, num=num),
        little_killer_cores(little_killers, num=num),
        fibonacci_line_cores(fibonacci_lines, num=num),
        arithmetic_sequence_line_cores(arithmetic_sequence_lines, num=num),
        lockout_line_cores(lockout_lines, min_endpoint_difference=4, num=num),
        region_sum_line_cores(region_sum_lines, num=num),
        inequality_cores(inequalities, num=num),
        parity_edge_cores(parity_edges, num=num),
        odd_even_cell_cores(odd_cells=odd_cells, even_cells=even_cells, num=num),
    )
    if include_initial_board and initial_board:
        start_assignment = {
            **initial_board_to_evidence(initial_board, num=num),
            **start_assignment,
        }
    if include_initial_board:
        return {
            **constraint_cores,
            **evidence_cores(start_assignment, constraint_cores),
        }
    return constraint_cores

