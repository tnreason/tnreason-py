"""
Sudoku Agents Module

This module contains agent-based constraint implementations and validation tests
for custom Sudoku rules (variant Sudoku constraints).

Available Constraints:
- knight_move: Anti-Knight constraint (knight's move positions cannot have same value)
- german_whispers: German Whispers constraint (adjacent digits differ by >= 5)
- renban: Renban constraint (line contains consecutive non-repeating digits)

Validation:
- validation/test_all_constraints.py: Tests validating constraint behavior.

Example Usage:
    from agents.constraints import (
        knight_move_constraint, generate_all_knight_constraints,
        german_whispers_constraint, generate_german_whispers_constraints,
        renban_constraint, generate_renban_constraints
    )
"""

from constraints.knight_move import (
    knight_move_constraint,
    cenc_knight_move_constraint,
    get_knight_moves_for_cell,
    generate_all_knight_constraints
)

from constraints.german_whispers import (
    german_whispers_constraint,
    cenc_german_whispers_constraint,
    generate_german_whispers_constraints
)

from constraints.renban import (
    renban_constraint,
    cenc_renban_constraint,
    generate_renban_constraints
)

__all__ = [
    'knight_move_constraint', 'cenc_knight_move_constraint',
    'get_knight_moves_for_cell', 'generate_all_knight_constraints',
    'german_whispers_constraint', 'cenc_german_whispers_constraint',
    'generate_german_whispers_constraints',
    'renban_constraint', 'cenc_renban_constraint',
    'generate_renban_constraints',
]
