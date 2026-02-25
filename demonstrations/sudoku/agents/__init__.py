"""
Sudoku Agents Module

This module contains agent-based constraint implementations and validation tests
for custom Sudoku rules (variant Sudoku constraints).

Available Constraints:
- knight_move: Anti-Knight constraint (knight's move positions cannot have same value)
- german_whispers: German Whispers constraint (adjacent digits on line differ by >= 5)
- renban: Renban constraint (line contains consecutive non-repeating digits)

Validation:
- validation/test_constraints.py: Comprehensive tests validating that each constraint
  correctly implements its natural language rule.

Example Usage:
    from agents.knight_move import knight_move_constraint, generate_all_knight_constraints
    from agents.german_whispers import german_whispers_constraint, generate_german_whispers_constraints
    from agents.renban import renban_constraint, generate_renban_constraints
    
    # Create a knight move constraint
    constraint = knight_move_constraint("r0c0", "r2c1")
    
    # Generate all knight constraints for a 9x9 grid
    all_knight_constraints = generate_all_knight_constraints()
    
    # Create a German whispers line constraint
    gw_line = [(0, 0), (0, 1), (0, 2)]
    gw_constraints = generate_german_whispers_constraints(gw_line, "gw_top_row")
    
    # Create a renban line constraint
    renban_line = [(0, 0), (1, 0), (2, 0)]
    renban_constraints = generate_renban_constraints(renban_line, "renban_first_col")
"""

from agents.knight_move import (
    knight_move_constraint,
    cenc_knight_move_constraint,
    get_knight_moves_for_cell,
    generate_all_knight_constraints
)

from agents.german_whispers import (
    german_whispers_constraint,
    cenc_german_whispers_constraint,
    generate_german_whispers_constraints
)

from agents.renban import (
    renban_constraint,
    cenc_renban_constraint,
    generate_renban_constraints
)

__all__ = [
    # Knight Move (Anti-Knight)
    'knight_move_constraint',
    'cenc_knight_move_constraint',
    'get_knight_moves_for_cell',
    'generate_all_knight_constraints',
    
    # German Whispers
    'german_whispers_constraint',
    'cenc_german_whispers_constraint',
    'generate_german_whispers_constraints',
    
    # Renban
    'renban_constraint',
    'cenc_renban_constraint',
    'generate_renban_constraints',
]
