"""
Killer Cage Constraint for Sudoku

Natural Language Rule: "Digits in cages sum to the small number printed in the corner."
                       "Digits inside Killer Cages cannot repeat, and must add up to 
                       the total indicated in the upper left corner of the cage."

A killer cage is a group of cells (usually outlined by a dotted line) with a small
number in the corner. The digits in the cage must:
1. Sum to the target value
2. Not repeat within the cage

Note: Some variations allow the sum to be "target ± 1" or have other modifications.
This implementation uses the standard rule (exact sum, no repeats).
"""

from tnreason import engine
from tnreason import representation
from itertools import permutations


def killer_cage_constraint(position_vars, target_sum, sudokuNum=3):
    """
    Creates a constraint that positions in a killer cage must contain
    non-repeating digits that sum to the target value.

    Args:
        position_vars: List of variable names for positions in the cage
        target_sum: Target sum for the cage (1-indexed, e.g., 10 means sum to 10)
        sudokuNum: Size parameter (default 3 for 9x9 sudoku, giving values 1-9)

    Returns:
        Dictionary with killer cage constraint tensor
    """
    n_positions = len(position_vars)
    max_digit = sudokuNum ** 2

    # Generate all valid assignments: non-repeating digits that sum to target
    valid_assignments = []

    def generate_valid_assignments(remaining_positions, available_values, current_sum, current_assignment):
        """Recursively generate all valid non-repeating assignments that sum to target."""
        if len(remaining_positions) == 0:
            # Check if we hit the target sum
            if current_sum == target_sum:
                valid_assignments.append((1, current_assignment.copy()))
            return

        # Pruning: check if it's still possible to reach target
        remaining_count = len(remaining_positions)
        min_possible_sum = current_sum + sum(sorted(available_values)[:remaining_count])
        max_possible_sum = current_sum + sum(sorted(available_values, reverse=True)[:remaining_count])

        if target_sum < min_possible_sum or target_sum > max_possible_sum:
            return  # Can't reach target with remaining values

        # Try each available value
        for i, val in enumerate(available_values):
            new_assignment = current_assignment.copy()
            new_assignment[position_vars[n_positions - len(remaining_positions)]] = val
            generate_valid_assignments(
                remaining_positions[1:],
                [v for j, v in enumerate(available_values) if j != i],
                current_sum + (val + 1),  # +1 because values are 0-indexed but sum is 1-indexed
                new_assignment
            )

    available_values = list(range(max_digit))  # 0-indexed values
    generate_valid_assignments(
        list(range(n_positions)),
        available_values,
        0,
        {}
    )

    shape = [sudokuNum ** 2] * n_positions

    return engine.create_from_slice_iterator(
        shape=shape,
        colors=position_vars,
        sliceIterator=valid_assignments
    )


def cenc_killer_cage_constraint(position_vars, target_sum, sudokuNum=3):
    """
    Creates a constraint that positions in a killer cage must contain
    non-repeating digits that sum to the target value.
    Uses the representation encoding approach.

    Args:
        position_vars: List of variable names for positions in the cage
        target_sum: Target sum for the cage (1-indexed)
        sudokuNum: Size parameter (default 3 for 9x9 sudoku)

    Returns:
        Tensor encoding for killer cage constraint
    """
    n_positions = len(position_vars)
    max_digit = sudokuNum ** 2

    def is_valid_killer_cage(*values):
        """Check if values are non-repeating and sum to target."""
        # Check for uniqueness
        if len(set(values)) != len(values):
            return 0

        # Check sum (values are 0-indexed, so add 1 to each)
        actual_sum = sum(v + 1 for v in values)
        if actual_sum != target_sum:
            return 0

        return 1

    shape = [max_digit] * n_positions

    return representation.create_tensor_encoding(
        inshape=shape,
        incolors=position_vars,
        function=is_valid_killer_cage
    )


def generate_killer_cage_constraints(cage_cells, cage_name="killer_cage"):
    """
    Generates a killer cage constraint for a cage of cells.

    Args:
        cage_cells: Tuple of (cells, target_sum) where cells is a list of (row, col) tuples
        cage_name: Name for the constraint

    Returns:
        Dictionary with killer cage constraint
    """
    if len(cage_cells) != 2:
        raise ValueError("cage_cells must be a tuple of (cells, target_sum)")

    cells, target_sum = cage_cells

    if len(cells) < 2:
        return {}

    position_vars = [f"r{row}c{col}" for row, col in cells]

    return {cage_name: cenc_killer_cage_constraint(position_vars, target_sum)}


def generate_killer_cages_multiple(cages):
    """
    Generates killer cage constraints for multiple cages.

    Args:
        cages: List of tuples ((cells, target_sum), cage_name)
               where cells is a list of (row, col) tuples

    Returns:
        Dictionary of all killer cage constraints
    """
    constraints = {}

    for cage_spec in cages:
        if len(cage_spec) == 2:
            (cells, target_sum), cage_name = cage_spec
        else:
            # Auto-generate name
            cells, target_sum = cage_spec
            cage_name = f"killer_cage_{len(constraints)}"

        cage_constraints = generate_killer_cage_constraints((cells, target_sum), cage_name)
        constraints.update(cage_constraints)

    return constraints


if __name__ == "__main__":
    # Test killer cage constraint with 2 positions, target sum 5
    pos_vars_2 = ["p1", "p2"]

    # Valid: non-repeating, sum to 5
    # (1,4), (4,1), (2,3), (3,2) in 1-indexed
    # (0,3), (3,0), (1,2), (2,1) in 0-indexed
    assert killer_cage_constraint(pos_vars_2, 5)[{"p1": 0, "p2": 3}] == 1  # 1+4=5
    assert killer_cage_constraint(pos_vars_2, 5)[{"p1": 3, "p2": 0}] == 1  # 4+1=5
    assert killer_cage_constraint(pos_vars_2, 5)[{"p1": 1, "p2": 2}] == 1  # 2+3=5
    assert killer_cage_constraint(pos_vars_2, 5)[{"p1": 2, "p2": 1}] == 1  # 3+2=5

    # Invalid: repeating
    assert killer_cage_constraint(pos_vars_2, 5)[{"p1": 1, "p2": 1}] == 0  # 2+2=4, also repeat

    # Invalid: wrong sum
    assert killer_cage_constraint(pos_vars_2, 5)[{"p1": 0, "p2": 0}] == 0  # 1+1=2
    assert killer_cage_constraint(pos_vars_2, 5)[{"p1": 0, "p2": 4}] == 0  # 1+5=6
    assert killer_cage_constraint(pos_vars_2, 5)[{"p1": 4, "p2": 4}] == 0  # 5+5=10

    # Test with 3 positions, target sum 6
    pos_vars_3 = ["p1", "p2", "p3"]
    # Valid: (1,2,3) and permutations -> (0,1,2) in 0-indexed
    assert killer_cage_constraint(pos_vars_3, 6)[{"p1": 0, "p2": 1, "p3": 2}] == 1
    assert killer_cage_constraint(pos_vars_3, 6)[{"p1": 2, "p2": 0, "p3": 1}] == 1  # different order

    # Invalid: repeating
    assert killer_cage_constraint(pos_vars_3, 6)[{"p1": 0, "p2": 0, "p3": 2}] == 0

    # Invalid: wrong sum
    assert killer_cage_constraint(pos_vars_3, 6)[{"p1": 0, "p2": 1, "p3": 3}] == 0  # 1+2+4=7

    # Test categorical encoding version
    assert cenc_killer_cage_constraint(pos_vars_2, 5)[{"p1": 0, "p2": 3}] == 1
    assert cenc_killer_cage_constraint(pos_vars_2, 5)[{"p1": 1, "p2": 2}] == 1
    assert cenc_killer_cage_constraint(pos_vars_2, 5)[{"p1": 0, "p2": 0}] == 0
    assert cenc_killer_cage_constraint(pos_vars_2, 5)[{"p1": 0, "p2": 4}] == 0

    # Test with larger cage (4 cells, target 10)
    pos_vars_4 = ["p1", "p2", "p3", "p4"]
    # Valid: (1,2,3,4) -> sum=10, (0,1,2,3) in 0-indexed
    assert cenc_killer_cage_constraint(pos_vars_4, 10)[{"p1": 0, "p2": 1, "p3": 2, "p4": 3}] == 1
    assert cenc_killer_cage_constraint(pos_vars_4, 10)[{"p1": 3, "p2": 2, "p3": 1, "p4": 0}] == 1

    # Invalid: repeating
    assert cenc_killer_cage_constraint(pos_vars_4, 10)[{"p1": 0, "p2": 0, "p3": 2, "p4": 3}] == 0

    # Invalid: wrong sum
    assert cenc_killer_cage_constraint(pos_vars_4, 10)[{"p1": 0, "p2": 1, "p3": 2, "p4": 4}] == 0  # sum=11

    # Test cage constraint generation
    cage_spec = ([(0, 0), (0, 1), (1, 0)], 7)  # 3 cells, sum to 7
    cage_constraints = generate_killer_cage_constraints(cage_spec, "test_cage")
    assert "test_cage" in cage_constraints

    # Test multiple cages - format: ((cells, target_sum), name)
    cages = [
        (([(0, 0), (0, 1)], 5), "cage1"),
        (([(1, 0), (1, 1), (1, 2)], 6), "cage2"),
    ]
    multi_constraints = generate_killer_cages_multiple(cages)
    assert "cage1" in multi_constraints
    assert "cage2" in multi_constraints

    print("All killer cage constraint tests passed!")
