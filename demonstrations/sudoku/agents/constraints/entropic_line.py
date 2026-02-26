"""
Entropic Line Constraint for Sudoku

Natural Language Rule: "Every string of three adjacent digits along a peach Entropic 
line must contain one low digit (123), one medium digit (456), and one high digit (789)."

An entropic line requires that any three consecutive cells on the line must contain:
- Exactly one digit from {1, 2, 3} (low)
- Exactly one digit from {4, 5, 6} (medium)
- Exactly one digit from {7, 8, 9} (high)

This ensures diversity in the digit distribution along the line.
"""

from tnreason import engine
from tnreason import representation
from itertools import permutations


def get_digit_category(digit_1indexed):
    """
    Returns the category of a digit (1-indexed).
    
    Args:
        digit_1indexed: Digit value (1-9)
    
    Returns:
        0 for low (1-3), 1 for medium (4-6), 2 for high (7-9)
    """
    if 1 <= digit_1indexed <= 3:
        return 0  # low
    elif 4 <= digit_1indexed <= 6:
        return 1  # medium
    else:  # 7-9
        return 2  # high


def entropic_line_constraint(position_vars, sudokuNum=3):
    """
    Creates a constraint that every three consecutive positions on an entropic line
    must contain one low (1-3), one medium (4-6), and one high (7-9) digit.

    Args:
        position_vars: List of variable names for positions on the entropic line
        sudokuNum: Size parameter (default 3 for 9x9 sudoku, giving values 1-9)

    Returns:
        Dictionary with entropic line constraint tensor
    """
    n_positions = len(position_vars)
    max_digit = sudokuNum ** 2

    if n_positions < 3:
        # For lines shorter than 3, no entropic constraint applies
        # Return a trivial constraint (all assignments valid)
        shape = [sudokuNum ** 2] * n_positions
        return engine.create_from_slice_iterator(
            shape=shape,
            colors=position_vars,
            sliceIterator=[(1, {position_vars[i]: val for i in range(n_positions)})
                          for val in range(max_digit)]
        )

    # Generate all valid assignments
    valid_assignments = []

    def is_entropic_valid(assignment):
        """Check if assignment satisfies entropic constraint for all 3-consecutive windows."""
        values = [assignment[var] for var in position_vars]

        # Check each window of 3 consecutive cells
        for i in range(len(values) - 2):
            window = values[i:i+3]
            categories = [get_digit_category(v + 1) for v in window]  # +1 for 1-indexed

            # Must have exactly one of each category
            if sorted(categories) != [0, 1, 2]:
                return False

        return True

    def generate_assignments(index, current_assignment):
        """Recursively generate all valid assignments."""
        if index == n_positions:
            if is_entropic_valid(current_assignment):
                valid_assignments.append((1, current_assignment.copy()))
            return

        # Try each possible value
        for val in range(max_digit):
            current_assignment[position_vars[index]] = val

            # Early pruning: check last 3-cell window if we have at least 3 values
            if index >= 2:
                last_three = [current_assignment[position_vars[i]] for i in range(index - 2, index + 1)]
                categories = [get_digit_category(v + 1) for v in last_three]
                if sorted(categories) != [0, 1, 2]:
                    continue  # Prune invalid branch

            generate_assignments(index + 1, current_assignment)

    generate_assignments(0, {})

    shape = [sudokuNum ** 2] * n_positions

    return engine.create_from_slice_iterator(
        shape=shape,
        colors=position_vars,
        sliceIterator=valid_assignments
    )


def cenc_entropic_line_constraint(position_vars, sudokuNum=3):
    """
    Creates a constraint that every three consecutive positions on an entropic line
    must contain one low (1-3), one medium (4-6), and one high (7-9) digit.
    Uses the representation encoding approach.

    Args:
        position_vars: List of variable names for positions on the entropic line
        sudokuNum: Size parameter (default 3 for 9x9 sudoku)

    Returns:
        Tensor encoding for entropic line constraint
    """
    n_positions = len(position_vars)
    max_digit = sudokuNum ** 2

    def is_entropic_valid(*values):
        """Check if values satisfy entropic constraint for all 3-consecutive windows."""
        if len(values) < 3:
            return 1  # No constraint for short lines

        # Check each window of 3 consecutive cells
        for i in range(len(values) - 2):
            window = values[i:i+3]
            categories = [get_digit_category(v + 1) for v in window]  # +1 for 1-indexed

            # Must have exactly one of each category
            if sorted(categories) != [0, 1, 2]:
                return 0

        return 1

    shape = [max_digit] * n_positions

    return representation.create_tensor_encoding(
        inshape=shape,
        incolors=position_vars,
        function=is_entropic_valid
    )


def generate_entropic_constraints(line_cells, line_name="entropic"):
    """
    Generates an entropic line constraint for a line of cells.

    Args:
        line_cells: List of (row, col) tuples defining the entropic line
        line_name: Name for the constraint

    Returns:
        Dictionary with entropic line constraint
    """
    if len(line_cells) < 3:
        return {}

    position_vars = [f"r{row}c{col}" for row, col in line_cells]

    return {line_name: cenc_entropic_line_constraint(position_vars)}


if __name__ == "__main__":
    # Test entropic line constraint with 3 positions
    pos_vars_3 = ["p1", "p2", "p3"]

    # Valid: one from each category
    # Low (1-3): 0,1,2 in 0-indexed
    # Medium (4-6): 3,4,5 in 0-indexed
    # High (7-9): 6,7,8 in 0-indexed
    assert entropic_line_constraint(pos_vars_3)[{"p1": 0, "p2": 3, "p3": 6}] == 1  # 1,4,7
    assert entropic_line_constraint(pos_vars_3)[{"p1": 2, "p2": 5, "p3": 8}] == 1  # 3,6,9
    assert entropic_line_constraint(pos_vars_3)[{"p1": 1, "p2": 4, "p3": 7}] == 1  # 2,5,8
    assert entropic_line_constraint(pos_vars_3)[{"p1": 0, "p2": 4, "p3": 8}] == 1  # 1,5,9

    # Different orderings are valid
    assert entropic_line_constraint(pos_vars_3)[{"p1": 6, "p2": 3, "p3": 0}] == 1  # 7,4,1
    assert entropic_line_constraint(pos_vars_3)[{"p1": 3, "p2": 0, "p3": 6}] == 1  # 4,1,7

    # Invalid: two from same category
    assert entropic_line_constraint(pos_vars_3)[{"p1": 0, "p2": 1, "p3": 6}] == 0  # 1,2,7 (two low)
    assert entropic_line_constraint(pos_vars_3)[{"p1": 0, "p2": 3, "p3": 4}] == 0  # 1,4,5 (two medium)
    assert entropic_line_constraint(pos_vars_3)[{"p1": 6, "p2": 7, "p3": 3}] == 0  # 7,8,4 (two high)
    assert entropic_line_constraint(pos_vars_3)[{"p1": 0, "p2": 0, "p3": 6}] == 0  # 1,1,7 (repeat low)

    # Test categorical encoding version
    assert cenc_entropic_line_constraint(pos_vars_3)[{"p1": 0, "p2": 3, "p3": 6}] == 1
    assert cenc_entropic_line_constraint(pos_vars_3)[{"p1": 2, "p2": 5, "p3": 8}] == 1
    assert cenc_entropic_line_constraint(pos_vars_3)[{"p1": 0, "p2": 1, "p3": 6}] == 0
    assert cenc_entropic_line_constraint(pos_vars_3)[{"p1": 6, "p2": 3, "p3": 0}] == 1

    # Test with 4 positions (two overlapping windows)
    pos_vars_4 = ["p1", "p2", "p3", "p4"]

    # Valid: both windows satisfy constraint
    # Window 1: p1,p2,p3 = 1,4,7 (low,med,high)
    # Window 2: p2,p3,p4 = 4,7,2 (med,high,low)
    assert cenc_entropic_line_constraint(pos_vars_4)[{"p1": 0, "p2": 3, "p3": 6, "p4": 1}] == 1

    # Valid: different pattern
    # Window 1: 3,6,9 (med,high,low) - wait, 3 is low (1-3), 6 is medium (4-6), 9 is high (7-9)
    # Let me recalculate: 3 (0-indexed 2) = low, 6 (0-indexed 5) = medium, 9 (0-indexed 8) = high
    # Window 1: p1,p2,p3 = 2,5,8 (low,med,high) -> 3,6,9
    # Window 2: p2,p3,p4 = 5,8,? need low -> 0,1,2
    assert cenc_entropic_line_constraint(pos_vars_4)[{"p1": 2, "p2": 5, "p3": 8, "p4": 0}] == 1

    # Invalid: second window fails
    # Window 1: 0,3,6 (low,med,high) OK
    # Window 2: 3,6,? need low but got medium (4)
    assert cenc_entropic_line_constraint(pos_vars_4)[{"p1": 0, "p2": 3, "p3": 6, "p4": 4}] == 0

    # Test with 2 positions (no constraint)
    pos_vars_2 = ["p1", "p2"]
    assert cenc_entropic_line_constraint(pos_vars_2)[{"p1": 0, "p2": 0}] == 1  # Any values OK
    assert cenc_entropic_line_constraint(pos_vars_2)[{"p1": 0, "p2": 8}] == 1  # Any values OK

    # Test line constraint generation
    line_cells = [(0, 0), (0, 1), (0, 2)]
    line_constraints = generate_entropic_constraints(line_cells, "test_entropic")
    assert "test_entropic" in line_constraints

    # Test short line (no constraint)
    short_line = [(0, 0), (0, 1)]
    short_constraints = generate_entropic_constraints(short_line, "short_entropic")
    assert len(short_constraints) == 0

    print("All entropic line constraint tests passed!")
