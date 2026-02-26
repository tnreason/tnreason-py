"""
Palindrome Constraint for Sudoku

Natural Language Rule: "Digits along a palindrome line read the same forwards and backwards."

A palindrome line requires that the sequence of digits is symmetric:
- First cell = Last cell
- Second cell = Second-to-last cell
- And so on...

Examples of valid palindrome lines:
- [1, 2, 1] - reads same forwards and backwards
- [3, 5, 7, 5, 3] - 5-cell palindrome
- [1, 1] - 2-cell palindrome (both same)
- [5] - single cell (trivially a palindrome)
- [1, 2, 3, 2, 1] - symmetric

Examples of invalid palindrome lines:
- [1, 2, 3] - not symmetric
- [1, 2, 2] - first != last
- [1, 2, 3, 4] - not symmetric
"""

from tnreason import engine
from tnreason import representation


def palindrome_constraint(position_vars, sudokuNum=3):
    """
    Creates a constraint that all positions on a palindrome line must
    read the same forwards and backwards.

    Args:
        position_vars: List of variable names for positions on the palindrome line
        sudokuNum: Size parameter (default 3 for 9x9 sudoku, giving values 1-9)

    Returns:
        Dictionary with palindrome constraint tensor
    """
    n_positions = len(position_vars)
    max_digit = sudokuNum ** 2

    # Generate all valid assignments (palindromic sequences)
    valid_assignments = []

    def generate_palindromes(index, current_assignment):
        """Recursively generate all palindromic assignments."""
        if index >= (n_positions + 1) // 2:
            # We've set all independent positions, mirror them
            full_assignment = current_assignment.copy()
            for i in range(n_positions // 2):
                mirror_idx = n_positions - 1 - i
                full_assignment[position_vars[mirror_idx]] = current_assignment[position_vars[i]]
            valid_assignments.append((1, full_assignment))
            return

        # Try each possible value for this position
        for val in range(max_digit):
            current_assignment[position_vars[index]] = val
            generate_palindromes(index + 1, current_assignment)

    generate_palindromes(0, {})

    shape = [sudokuNum ** 2] * n_positions

    return engine.create_from_slice_iterator(
        shape=shape,
        colors=position_vars,
        sliceIterator=valid_assignments
    )


def cenc_palindrome_constraint(position_vars, sudokuNum=3):
    """
    Creates a constraint that all positions on a palindrome line must
    read the same forwards and backwards.
    Uses the representation encoding approach.

    Args:
        position_vars: List of variable names for positions on the palindrome line
        sudokuNum: Size parameter (default 3 for 9x9 sudoku)

    Returns:
        Tensor encoding for palindrome constraint
    """
    n_positions = len(position_vars)
    max_digit = sudokuNum ** 2

    def is_palindrome(*values):
        """Check if values form a palindrome."""
        for i in range(len(values) // 2):
            if values[i] != values[len(values) - 1 - i]:
                return 0
        return 1

    shape = [max_digit] * n_positions

    return representation.create_tensor_encoding(
        inshape=shape,
        incolors=position_vars,
        function=is_palindrome
    )


def generate_palindrome_constraints(line_cells, line_name="palindrome"):
    """
    Generates a palindrome constraint for a line of cells.

    Args:
        line_cells: List of (row, col) tuples defining the palindrome line
        line_name: Name for the constraint

    Returns:
        Dictionary with palindrome constraint
    """
    if len(line_cells) < 2:
        return {}

    position_vars = [f"r{row}c{col}" for row, col in line_cells]

    return {line_name: cenc_palindrome_constraint(position_vars)}


if __name__ == "__main__":
    # Test palindrome constraint with 3 positions
    pos_vars_3 = ["p1", "p2", "p3"]

    # Valid: palindromic sequences
    assert palindrome_constraint(pos_vars_3)[{"p1": 0, "p2": 1, "p3": 0}] == 1  # 1,2,1
    assert palindrome_constraint(pos_vars_3)[{"p1": 2, "p2": 4, "p3": 2}] == 1  # 3,5,3
    assert palindrome_constraint(pos_vars_3)[{"p1": 0, "p2": 0, "p3": 0}] == 1  # 1,1,1
    assert palindrome_constraint(pos_vars_3)[{"p1": 8, "p2": 4, "p3": 8}] == 1  # 9,5,9

    # Invalid: not palindromic
    assert palindrome_constraint(pos_vars_3)[{"p1": 0, "p2": 1, "p3": 2}] == 0  # 1,2,3
    assert palindrome_constraint(pos_vars_3)[{"p1": 0, "p2": 0, "p3": 1}] == 0  # 1,1,2
    assert palindrome_constraint(pos_vars_3)[{"p1": 1, "p2": 2, "p3": 3}] == 0  # 2,3,4

    # Test categorical encoding version
    assert cenc_palindrome_constraint(pos_vars_3)[{"p1": 0, "p2": 1, "p3": 0}] == 1
    assert cenc_palindrome_constraint(pos_vars_3)[{"p1": 2, "p2": 4, "p3": 2}] == 1
    assert cenc_palindrome_constraint(pos_vars_3)[{"p1": 0, "p2": 1, "p3": 2}] == 0

    # Test with 2 positions
    pos_vars_2 = ["p1", "p2"]
    assert cenc_palindrome_constraint(pos_vars_2)[{"p1": 0, "p2": 0}] == 1  # 1,1
    assert cenc_palindrome_constraint(pos_vars_2)[{"p1": 5, "p2": 5}] == 1  # 6,6
    assert cenc_palindrome_constraint(pos_vars_2)[{"p1": 0, "p2": 1}] == 0  # 1,2 not palindrome

    # Test with 4 positions
    pos_vars_4 = ["p1", "p2", "p3", "p4"]
    assert cenc_palindrome_constraint(pos_vars_4)[{"p1": 0, "p2": 1, "p3": 1, "p4": 0}] == 1  # 1,2,2,1
    assert cenc_palindrome_constraint(pos_vars_4)[{"p1": 2, "p2": 4, "p3": 6, "p4": 2}] == 0  # 3,5,7,3 not palindrome
    assert cenc_palindrome_constraint(pos_vars_4)[{"p1": 2, "p2": 4, "p3": 4, "p4": 2}] == 1  # 3,5,5,3
    assert cenc_palindrome_constraint(pos_vars_4)[{"p1": 0, "p2": 4, "p3": 4, "p4": 0}] == 1  # 1,5,5,1

    # Test with 5 positions
    pos_vars_5 = ["p1", "p2", "p3", "p4", "p5"]
    assert cenc_palindrome_constraint(pos_vars_5)[{"p1": 0, "p2": 1, "p3": 2, "p4": 1, "p5": 0}] == 1  # 1,2,3,2,1
    assert cenc_palindrome_constraint(pos_vars_5)[{"p1": 2, "p2": 4, "p3": 6, "p4": 4, "p5": 2}] == 1  # 3,5,7,5,3
    assert cenc_palindrome_constraint(pos_vars_5)[{"p1": 0, "p2": 1, "p3": 2, "p4": 3, "p5": 4}] == 0  # 1,2,3,4,5 not palindrome

    # Test single position (trivially a palindrome)
    pos_vars_1 = ["p1"]
    assert cenc_palindrome_constraint(pos_vars_1)[{"p1": 0}] == 1
    assert cenc_palindrome_constraint(pos_vars_1)[{"p1": 5}] == 1

    # Test line constraint generation
    line_cells = [(0, 0), (0, 1), (0, 2)]
    line_constraints = generate_palindrome_constraints(line_cells, "test_palindrome")
    assert "test_palindrome" in line_constraints

    print("All palindrome constraint tests passed!")
