"""
Arrow Constraint for Sudoku

Natural Language Rule: "Digits along an arrow sum to the digit in the corresponding circle."
                       "A digit in a circle with an arrow attached is equal to the sum 
                       of the digits on the attached arrow."

An arrow consists of:
- A circle cell containing a single digit
- Arrow cells (line of cells extending from the circle)
- The digit in the circle must equal the sum of all digits on the arrow

Note: Arrow cells CAN repeat digits (unlike killer cages).
"""

from tnreason import engine
from tnreason import representation


def arrow_constraint(circle_var, arrow_vars, sudokuNum=3):
    """
    Creates a constraint that the circle position must contain a digit
    equal to the sum of all digits on the arrow.

    Args:
        circle_var: Variable name for the circle position
        arrow_vars: List of variable names for positions on the arrow
        sudokuNum: Size parameter (default 3 for 9x9 sudoku, giving values 1-9)

    Returns:
        Dictionary with arrow constraint tensor
    """
    n_arrow_cells = len(arrow_vars)
    max_digit = sudokuNum ** 2

    # Generate all valid assignments where circle = sum of arrow values
    valid_assignments = []

    def generate_valid_assignments(arrow_index, arrow_values, current_sum):
        """Recursively generate all valid arrow assignments."""
        if arrow_index == n_arrow_cells:
            # Check if sum is a valid digit (1-9)
            if 1 <= current_sum <= max_digit:
                assignment = {circle_var: current_sum - 1}  # -1 for 0-indexed
                for i, val in enumerate(arrow_values):
                    assignment[arrow_vars[i]] = val
                valid_assignments.append((1, assignment))
            return

        # Try each possible value for this arrow cell
        for val in range(max_digit):
            new_sum = current_sum + (val + 1)  # +1 because values are 0-indexed
            # Pruning: if sum already exceeds max_digit, stop
            if new_sum <= max_digit:
                generate_valid_assignments(
                    arrow_index + 1,
                    arrow_values + [val],
                    new_sum
                )

    generate_valid_assignments(0, [], 0)

    # Build shape: circle + all arrow cells
    all_vars = [circle_var] + arrow_vars
    shape = [sudokuNum ** 2] * len(all_vars)

    return engine.create_from_slice_iterator(
        shape=shape,
        colors=all_vars,
        sliceIterator=valid_assignments
    )


def cenc_arrow_constraint(circle_var, arrow_vars, sudokuNum=3):
    """
    Creates a constraint that the circle position must contain a digit
    equal to the sum of all digits on the arrow.
    Uses the representation encoding approach.

    Args:
        circle_var: Variable name for the circle position
        arrow_vars: List of variable names for positions on the arrow
        sudokuNum: Size parameter (default 3 for 9x9 sudoku)

    Returns:
        Tensor encoding for arrow constraint
    """
    n_arrow_cells = len(arrow_vars)
    max_digit = sudokuNum ** 2

    def is_valid_arrow_func(*values):
        """Check if circle value equals sum of arrow values."""
        circle_value = values[0] + 1  # Convert to 1-indexed
        arrow_values = [v + 1 for v in values[1:]]  # Convert to 1-indexed
        arrow_sum = sum(arrow_values)

        return 1 if circle_value == arrow_sum else 0

    # Build shape: circle + all arrow cells
    all_vars = [circle_var] + arrow_vars
    shape = [max_digit] * len(all_vars)

    return representation.create_tensor_encoding(
        inshape=shape,
        incolors=all_vars,
        function=is_valid_arrow_func
    )


def generate_arrow_constraints(circle_cell, arrow_cells, arrow_name="arrow"):
    """
    Generates an arrow constraint for a circle and arrow cells.

    Args:
        circle_cell: (row, col) tuple for the circle position
        arrow_cells: List of (row, col) tuples for arrow positions
        arrow_name: Name for the constraint

    Returns:
        Dictionary with arrow constraint
    """
    if len(arrow_cells) < 1:
        return {}

    circle_var = f"r{circle_cell[0]}c{circle_cell[1]}"
    arrow_vars = [f"r{row}c{col}" for row, col in arrow_cells]

    return {arrow_name: cenc_arrow_constraint(circle_var, arrow_vars)}


def generate_arrows_multiple(arrows):
    """
    Generates arrow constraints for multiple arrows.

    Args:
        arrows: List of tuples (circle_cell, arrow_cells, arrow_name)
                where circle_cell and arrow_cells are (row, col) tuples

    Returns:
        Dictionary of all arrow constraints
    """
    constraints = {}

    for arrow_spec in arrows:
        if len(arrow_spec) == 3:
            circle_cell, arrow_cells, arrow_name = arrow_spec
        else:
            # Auto-generate name
            circle_cell, arrow_cells = arrow_spec
            arrow_name = f"arrow_{len(constraints)}"

        arrow_constraints = generate_arrow_constraints(circle_cell, arrow_cells, arrow_name)
        constraints.update(arrow_constraints)

    return constraints


if __name__ == "__main__":
    # Test arrow constraint with 1 arrow cell
    circle = "circle"
    arrow1 = ["arrow1"]

    # Valid: circle (1-idx) = arrow (1-idx), so circle (0-idx) = arrow (0-idx)
    assert arrow_constraint(circle, arrow1)[{circle: 0, arrow1[0]: 0}] == 1  # 1 = 1
    assert arrow_constraint(circle, arrow1)[{circle: 1, arrow1[0]: 1}] == 1  # 2 = 2
    assert arrow_constraint(circle, arrow1)[{circle: 8, arrow1[0]: 8}] == 1  # 9 = 9

    # Invalid: circle != arrow value
    assert arrow_constraint(circle, arrow1)[{circle: 1, arrow1[0]: 0}] == 0  # 2 != 1
    assert arrow_constraint(circle, arrow1)[{circle: 5, arrow1[0]: 0}] == 0  # 6 != 1

    # Test with 2 arrow cells
    arrow2 = ["a1", "a2"]

    # Valid: circle (1-idx) = sum of arrows (1-idx)
    assert arrow_constraint(circle, arrow2)[{circle: 2, "a1": 0, "a2": 1}] == 1  # 3 = 1+2
    assert arrow_constraint(circle, arrow2)[{circle: 2, "a1": 1, "a2": 0}] == 1  # 3 = 2+1
    assert arrow_constraint(circle, arrow2)[{circle: 4, "a1": 1, "a2": 2}] == 1  # 5 = 2+3
    assert arrow_constraint(circle, arrow2)[{circle: 8, "a1": 3, "a2": 4}] == 1  # 9 = 4+5
    assert arrow_constraint(circle, arrow2)[{circle: 1, "a1": 0, "a2": 0}] == 1  # 2 = 1+1

    # Invalid
    assert arrow_constraint(circle, arrow2)[{circle: 0, "a1": 0, "a2": 0}] == 0  # 1 != 1+1=2
    assert arrow_constraint(circle, arrow2)[{circle: 5, "a1": 2, "a2": 3}] == 0  # 6 != 3+4=7

    # Test categorical encoding version
    assert cenc_arrow_constraint(circle, arrow1)[{circle: 0, arrow1[0]: 0}] == 1
    assert cenc_arrow_constraint(circle, arrow1)[{circle: 1, arrow1[0]: 1}] == 1
    assert cenc_arrow_constraint(circle, arrow1)[{circle: 1, arrow1[0]: 0}] == 0

    assert cenc_arrow_constraint(circle, arrow2)[{circle: 2, "a1": 0, "a2": 1}] == 1
    assert cenc_arrow_constraint(circle, arrow2)[{circle: 4, "a1": 1, "a2": 2}] == 1
    assert cenc_arrow_constraint(circle, arrow2)[{circle: 0, "a1": 0, "a2": 0}] == 0

    # Test with 3 arrow cells
    arrow3 = ["a1", "a2", "a3"]

    # Valid: circle = sum of 3 arrows
    assert cenc_arrow_constraint(circle, arrow3)[{circle: 5, "a1": 0, "a2": 1, "a3": 2}] == 1  # 6 = 1+2+3
    assert cenc_arrow_constraint(circle, arrow3)[{circle: 8, "a1": 1, "a2": 2, "a3": 3}] == 1  # 9 = 2+3+4

    # Invalid
    assert cenc_arrow_constraint(circle, arrow3)[{circle: 5, "a1": 0, "a2": 0, "a3": 0}] == 0  # 6 != 1+1+1=3

    # Test arrow constraint generation
    circle_cell = (0, 0)
    arrow_cells = [(0, 1), (0, 2)]
    arrow_constraints = generate_arrow_constraints(circle_cell, arrow_cells, "test_arrow")
    assert "test_arrow" in arrow_constraints

    # Test multiple arrows
    arrows = [
        ((0, 0), [(0, 1), (0, 2)], "arrow1"),
        ((1, 0), [(1, 1)], "arrow2"),
    ]
    multi_constraints = generate_arrows_multiple(arrows)
    assert "arrow1" in multi_constraints
    assert "arrow2" in multi_constraints

    print("All arrow constraint tests passed!")
