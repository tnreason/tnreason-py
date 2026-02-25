"""
Sudoku Variant Constraints

This package contains implementations of custom Sudoku variant rules:
- knight_move: Anti-Knight (knight's move positions cannot have same value)
- german_whispers: German Whispers (adjacent digits differ by >= 5)
- renban: Renban (consecutive non-repeating digits on a line)
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
