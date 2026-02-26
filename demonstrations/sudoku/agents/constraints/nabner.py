"""
Nabner Line Constraint for Sudoku

Natural Language Rule: "The set of digits on an orange nabner line must not repeat and 
                       no two digits on the line may be consecutive."

A nabner line requires:
1. All digits must be unique (non-repeating)
2. No two digits can be consecutive (|v1 - v2| > 1 for all pairs)

This is stricter than just having non-consecutive adjacent cells - 
NO two digits anywhere on the line can be consecutive.

Examples of valid nabner lines:
- [1, 3, 5] - all unique, no consecutive pairs
- [1, 4, 7] - all unique, gaps of at least 2
- [2, 5, 8] - all unique, gaps of at least 2

Examples of invalid nabner lines:
- [1, 2, 4] - 1 and 2 are consecutive
- [1, 3, 3] - 3 repeats
- [2, 4, 5] - 4 and 5 are consecutive
"""

from tnreason import engine
from tnreason import representation


def nabner_constraint(position_vars, sudokuNum=3):
    """
    Creates a constraint that all positions on a nabner line must contain
    unique digits where no two digits are consecutive.

    Args:
        position_vars: List of variable names for positions on the nabner line
        sudokuNum: Size parameter (default 3 for 9x9 sudoku, giving values 1-9)

    Returns:
        Dictionary with nabner constraint tensor
    """
    n_positions = len(position_vars)
    max_digit = sudokuNum ** 2

    # Generate all valid assignments
    valid_assignments = []

    def is_valid_nabner(assignment):
        """Check if assignment satisfies nabner constraints."""
        values = [assignment[var] for var in position_vars]
        
        # Check uniqueness
        if len(set(values)) != len(values):
            return False
        
        # Check no two values are consecutive
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if abs(values[i] - values[j]) == 1:
                    return False
        
        return True

    def generate_assignments(index, current_assignment):
        """Recursively generate all valid assignments."""
        if index == n_positions:
            if is_valid_nabner(current_assignment):
                valid_assignments.append((1, current_assignment.copy()))
            return

        # Try each possible value
        for val in range(max_digit):
            # Early pruning: check if this value conflicts with any previous
            conflict = False
            for prev_idx in range(index):
                prev_val = current_assignment[position_vars[prev_idx]]
                # Check uniqueness
                if prev_val == val:
                    conflict = True
                    break
                # Check consecutive
                if abs(prev_val - val) == 1:
                    conflict = True
                    break
            
            if not conflict:
                current_assignment[position_vars[index]] = val
                generate_assignments(index + 1, current_assignment)

    generate_assignments(0, {})

    shape = [sudokuNum ** 2] * n_positions

    return engine.create_from_slice_iterator(
        shape=shape,
        colors=position_vars,
        sliceIterator=valid_assignments
    )


def cenc_nabner_constraint(position_vars, sudokuNum=3):
    """
    Creates a constraint that all positions on a nabner line must contain
    unique digits where no two digits are consecutive.
    Uses the representation encoding approach.

    Args:
        position_vars: List of variable names for positions on the nabner line
        sudokuNum: Size parameter (default 3 for 9x9 sudoku)

    Returns:
        Tensor encoding for nabner constraint
    """
    n_positions = len(position_vars)
    max_digit = sudokuNum ** 2

    def is_valid_nabner(*values):
        """Check if values satisfy nabner constraints."""
        # Check uniqueness
        if len(set(values)) != len(values):
            return 0
        
        # Check no two values are consecutive
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if abs(values[i] - values[j]) == 1:
                    return 0
        
        return 1

    shape = [max_digit] * n_positions

    return representation.create_tensor_encoding(
        inshape=shape,
        incolors=position_vars,
        function=is_valid_nabner
    )


def generate_nabner_constraints(line_cells, line_name="nabner"):
    """
    Generates a nabner line constraint for a line of cells.

    Args:
        line_cells: List of (row, col) tuples defining the nabner line
        line_name: Name for the constraint

    Returns:
        Dictionary with nabner line constraint
    """
    if len(line_cells) < 2:
        return {}

    position_vars = [f"r{row}c{col}" for row, col in line_cells]

    return {line_name: cenc_nabner_constraint(position_vars)}


if __name__ == "__main__":
    # Test nabner constraint with 3 positions
    pos_vars_3 = ["p1", "p2", "p3"]

    # Valid: unique, no consecutive pairs
    assert nabner_constraint(pos_vars_3)[{"p1": 0, "p2": 2, "p3": 4}] == 1  # 1,3,5
    assert nabner_constraint(pos_vars_3)[{"p1": 0, "p2": 3, "p3": 6}] == 1  # 1,4,7
    assert nabner_constraint(pos_vars_3)[{"p1": 1, "p2": 4, "p3": 7}] == 1  # 2,5,8
    assert nabner_constraint(pos_vars_3)[{"p1": 4, "p2": 2, "p3": 0}] == 1  # 5,3,1 (different order)

    # Invalid: has consecutive pair
    assert nabner_constraint(pos_vars_3)[{"p1": 0, "p2": 1, "p3": 3}] == 0  # 1,2,4 (1,2 consecutive)
    assert nabner_constraint(pos_vars_3)[{"p1": 0, "p2": 2, "p3": 3}] == 0  # 1,3,4 (3,4 consecutive)
    assert nabner_constraint(pos_vars_3)[{"p1": 3, "p2": 4, "p3": 6}] == 0  # 4,5,7 (4,5 consecutive)

    # Invalid: repeating
    assert nabner_constraint(pos_vars_3)[{"p1": 0, "p2": 0, "p3": 2}] == 0  # 1,1,3
    assert nabner_constraint(pos_vars_3)[{"p1": 2, "p2": 4, "p3": 2}] == 0  # 3,5,3

    # Test categorical encoding version
    assert cenc_nabner_constraint(pos_vars_3)[{"p1": 0, "p2": 2, "p3": 4}] == 1
    assert cenc_nabner_constraint(pos_vars_3)[{"p1": 0, "p2": 3, "p3": 6}] == 1
    assert cenc_nabner_constraint(pos_vars_3)[{"p1": 0, "p2": 1, "p3": 3}] == 0
    assert cenc_nabner_constraint(pos_vars_3)[{"p1": 0, "p2": 0, "p3": 2}] == 0

    # Test with 2 positions
    pos_vars_2 = ["p1", "p2"]
    assert cenc_nabner_constraint(pos_vars_2)[{"p1": 0, "p2": 2}] == 1  # 1,3 OK
    assert cenc_nabner_constraint(pos_vars_2)[{"p1": 0, "p2": 1}] == 0  # 1,2 consecutive
    assert cenc_nabner_constraint(pos_vars_2)[{"p1": 0, "p2": 0}] == 0  # repeat

    # Test with 4 positions
    pos_vars_4 = ["p1", "p2", "p3", "p4"]
    # Valid: 1,3,5,7 or 1,3,5,8 or 1,3,6,8 or 1,4,6,8 or 2,4,6,8
    assert cenc_nabner_constraint(pos_vars_4)[{"p1": 0, "p2": 2, "p3": 4, "p4": 6}] == 1  # 1,3,5,7
    assert cenc_nabner_constraint(pos_vars_4)[{"p1": 1, "p2": 3, "p3": 5, "p4": 7}] == 1  # 2,4,6,8
    # Invalid: has consecutive
    assert cenc_nabner_constraint(pos_vars_4)[{"p1": 0, "p2": 2, "p3": 4, "p4": 5}] == 0  # 1,3,5,6 (5,6 consecutive)

    # Test line constraint generation
    line_cells = [(0, 0), (0, 1), (0, 2)]
    line_constraints = generate_nabner_constraints(line_cells, "test_nabner")
    assert "test_nabner" in line_constraints

    print("All nabner constraint tests passed!")
