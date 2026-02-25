"""
Renban Constraint for Sudoku

Natural Language Rule: "Each purple line must contain a set of consecutive non-repeating digits in any order."
                       "Digits on a purple renban line are consecutive and non-repeating."
                       "Digits along a purple line must form a set of non-repeating consecutive digits (in any order)."

This means that all digits on a renban line must:
1. Be unique (non-repeating)
2. Form a consecutive range (e.g., {2,3,4,5} or {5,6,7}, but not {1,3,5} or {2,2,3,4})

Examples of valid renban lines:
- [1, 2, 3] - consecutive, non-repeating
- [3, 1, 2] - same set, different order (order doesn't matter)
- [5, 6, 7, 8] - consecutive range of 4 digits
- [9, 7, 8] - consecutive range {7,8,9} in different order

Examples of invalid renban lines:
- [1, 2, 4] - not consecutive (missing 3)
- [1, 2, 2, 3] - repeating digit (2)
- [1, 3, 5] - not consecutive
- [5, 5, 6, 7] - repeating digit (5)
"""

from tnreason import engine
from tnreason import representation
from itertools import permutations


def renban_constraint(position_vars, sudokuNum=3):
    """
    Creates a constraint that all positions on a renban line must contain
    consecutive non-repeating digits.
    
    Args:
        position_vars: List of variable names for positions on the renban line
        sudokuNum: Size parameter (default 3 for 9x9 sudoku, giving values 1-9)
    
    Returns:
        Dictionary with renban constraint tensor
    """
    n_positions = len(position_vars)
    max_digit = sudokuNum ** 2
    
    # Generate all valid assignments: consecutive non-repeating digits
    valid_assignments = []
    
    # For each possible starting digit of the consecutive range
    for start in range(max_digit - n_positions + 1):
        # The consecutive range is [start, start+1, ..., start+n_positions-1]
        consecutive_range = list(range(start, start + n_positions))
        
        # All permutations of this range are valid
        for perm in permutations(consecutive_range):
            assignment = {position_vars[i]: perm[i] for i in range(n_positions)}
            valid_assignments.append((1, assignment))
    
    shape = [sudokuNum ** 2] * n_positions
    
    return engine.create_from_slice_iterator(
        shape=shape,
        colors=position_vars,
        sliceIterator=valid_assignments
    )


def cenc_renban_constraint(position_vars, sudokuNum=3):
    """
    Creates a constraint that all positions on a renban line must contain
    consecutive non-repeating digits.
    Uses the representation encoding approach.
    
    Args:
        position_vars: List of variable names for positions on the renban line
        sudokuNum: Size parameter (default 3 for 9x9 sudoku)
    
    Returns:
        Tensor encoding for renban constraint
    """
    n_positions = len(position_vars)
    max_digit = sudokuNum ** 2
    
    def is_valid_renban(*values):
        """Check if values form a consecutive non-repeating set."""
        # Check for uniqueness
        if len(set(values)) != len(values):
            return 0
        
        # Check if consecutive
        min_val = min(values)
        max_val = max(values)
        
        # For consecutive values: max - min == len - 1
        if max_val - min_val != len(values) - 1:
            return 0
        
        return 1
    
    shape = [max_digit] * n_positions
    
    return representation.create_tensor_encoding(
        inshape=shape,
        incolors=position_vars,
        function=is_valid_renban
    )


def generate_renban_constraints(line_cells, line_name="renban"):
    """
    Generates a renban constraint for a line of cells.
    
    Args:
        line_cells: List of (row, col) tuples defining the renban line
        line_name: Name for the constraint
    
    Returns:
        Dictionary with renban constraint
    """
    if len(line_cells) < 2:
        return {}
    
    position_vars = [f"r{row}c{col}" for row, col in line_cells]
    
    return {line_name: cenc_renban_constraint(position_vars)}


if __name__ == "__main__":
    # Test renban constraint with 3 positions
    pos_vars = ["p1", "p2", "p3"]
    
    # Valid: consecutive non-repeating (1,2,3) -> indices (0,1,2)
    assert renban_constraint(pos_vars)[{"p1": 0, "p2": 1, "p3": 2}] == 1
    assert renban_constraint(pos_vars)[{"p1": 2, "p2": 0, "p3": 1}] == 1  # different order
    assert renban_constraint(pos_vars)[{"p1": 1, "p2": 2, "p3": 3}] == 1  # (2,3,4)
    
    # Invalid: not consecutive
    assert renban_constraint(pos_vars)[{"p1": 0, "p2": 1, "p3": 3}] == 0  # missing 2
    assert renban_constraint(pos_vars)[{"p1": 0, "p2": 2, "p3": 4}] == 0  # gaps
    
    # Invalid: repeating
    assert renban_constraint(pos_vars)[{"p1": 1, "p2": 1, "p3": 2}] == 0
    assert renban_constraint(pos_vars)[{"p1": 2, "p2": 2, "p3": 2}] == 0
    
    # Test categorical encoding version
    assert cenc_renban_constraint(pos_vars)[{"p1": 0, "p2": 1, "p3": 2}] == 1
    assert cenc_renban_constraint(pos_vars)[{"p1": 2, "p2": 0, "p3": 1}] == 1
    assert cenc_renban_constraint(pos_vars)[{"p1": 0, "p2": 1, "p3": 3}] == 0
    assert cenc_renban_constraint(pos_vars)[{"p1": 1, "p2": 1, "p3": 2}] == 0
    
    # Test with 2 positions (simpler case)
    pos_vars_2 = ["p1", "p2"]
    assert renban_constraint(pos_vars_2)[{"p1": 0, "p2": 1}] == 1  # (1,2)
    assert renban_constraint(pos_vars_2)[{"p1": 1, "p2": 0}] == 1  # (2,1)
    assert renban_constraint(pos_vars_2)[{"p1": 0, "p2": 2}] == 0  # (1,3) - not consecutive
    assert renban_constraint(pos_vars_2)[{"p1": 0, "p2": 0}] == 0  # same value
    
    # Test with 4 positions
    pos_vars_4 = ["p1", "p2", "p3", "p4"]
    assert renban_constraint(pos_vars_4)[{"p1": 0, "p2": 1, "p3": 2, "p4": 3}] == 1  # (1,2,3,4)
    assert renban_constraint(pos_vars_4)[{"p1": 3, "p2": 1, "p3": 2, "p4": 0}] == 1  # shuffled
    assert renban_constraint(pos_vars_4)[{"p1": 5, "p2": 6, "p3": 7, "p4": 8}] == 1  # (6,7,8,9)
    assert renban_constraint(pos_vars_4)[{"p1": 0, "p2": 1, "p3": 2, "p4": 4}] == 0  # gap
    
    # Test line constraint generation
    line = [(0, 0), (0, 1), (0, 2)]
    line_constraints = generate_renban_constraints(line, "test_renban")
    assert "test_renban" in line_constraints
    
    print("All renban constraint tests passed!")
