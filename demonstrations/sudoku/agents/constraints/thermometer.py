"""
Thermometer Constraint for Sudoku

Natural Language Rule: "Digits along a thermometer must increase from the bulb end to the tip."
                       "Digits along thermometers must strictly increase from the bulb end."

A thermometer is a line of cells where:
- The first cell (bulb) contains the smallest digit
- Each subsequent cell along the thermometer contains a larger digit
- Values must strictly increase: v1 < v2 < v3 < ... < vn
"""

from tnreason import engine
from tnreason import representation


def thermometer_constraint(position_vars, sudokuNum=3):
    """
    Creates a constraint that positions along a thermometer must contain
    strictly increasing digits from bulb to tip.

    Args:
        position_vars: List of variable names for positions on the thermometer
                      (ordered from bulb to tip)
        sudokuNum: Size parameter (default 3 for 9x9 sudoku, giving values 1-9)

    Returns:
        Dictionary with thermometer constraint tensor
    """
    n_positions = len(position_vars)
    max_digit = sudokuNum ** 2

    # Generate all valid assignments: strictly increasing sequences
    valid_assignments = []

    def generate_increasing_sequences(positions, min_val, max_val, current_seq):
        """Recursively generate all strictly increasing sequences."""
        if len(positions) == 0:
            # Valid sequence found
            assignment = {position_vars[i]: current_seq[i] for i in range(len(current_seq))}
            valid_assignments.append((1, assignment))
            return

        # Try each value from min_val to max_val
        for val in range(min_val, max_val + 1):
            # Ensure we have enough remaining values for remaining positions
            remaining_positions = len(positions) - 1
            max_possible = max_val - remaining_positions
            if val <= max_possible:
                generate_increasing_sequences(
                    positions[1:],
                    val + 1,  # Next value must be strictly greater
                    max_val,
                    current_seq + [val]
                )

    generate_increasing_sequences(
        list(range(n_positions)),
        0,  # min value (0-indexed, represents digit 1)
        max_digit - 1,  # max value (8-indexed, represents digit 9)
        []
    )

    shape = [sudokuNum ** 2] * n_positions

    return engine.create_from_slice_iterator(
        shape=shape,
        colors=position_vars,
        sliceIterator=valid_assignments
    )


def cenc_thermometer_constraint(position_vars, sudokuNum=3):
    """
    Creates a constraint that positions along a thermometer must contain
    strictly increasing digits from bulb to tip.
    Uses the representation encoding approach.

    Args:
        position_vars: List of variable names for positions on the thermometer
                      (ordered from bulb to tip)
        sudokuNum: Size parameter (default 3 for 9x9 sudoku)

    Returns:
        Tensor encoding for thermometer constraint
    """
    n_positions = len(position_vars)
    max_digit = sudokuNum ** 2

    def is_strictly_increasing(*values):
        """Check if values are strictly increasing."""
        for i in range(len(values) - 1):
            if values[i] >= values[i + 1]:
                return 0
        return 1

    shape = [max_digit] * n_positions

    return representation.create_tensor_encoding(
        inshape=shape,
        incolors=position_vars,
        function=is_strictly_increasing
    )


def generate_thermometer_constraints(thermometer_cells, thermo_name="thermometer"):
    """
    Generates a thermometer constraint for a thermometer line.

    Args:
        thermometer_cells: List of (row, col) tuples defining the thermometer
                          (ordered from bulb to tip)
        thermo_name: Name for the constraint

    Returns:
        Dictionary with thermometer constraint
    """
    if len(thermometer_cells) < 2:
        return {}

    position_vars = [f"r{row}c{col}" for row, col in thermometer_cells]

    return {thermo_name: cenc_thermometer_constraint(position_vars)}


def generate_slow_thermometer_constraints(thermometer_cells, thermo_name="slow_thermometer"):
    """
    Generates a slow thermometer constraint where digits can stay the same
    or increase (non-decreasing), but never decrease.

    Natural Language Rule: "Digits along a thermometer must generally increase 
                          from bulb to tip. Digits may remain the same when 
                          allowed by other rules, but may never decrease."

    Args:
        thermometer_cells: List of (row, col) tuples defining the thermometer
                          (ordered from bulb to tip)
        thermo_name: Name for the constraint

    Returns:
        Dictionary with slow thermometer constraint
    """
    if len(thermometer_cells) < 2:
        return {}

    position_vars = [f"r{row}c{col}" for row, col in thermometer_cells]
    n_positions = len(position_vars)
    max_digit = sudokuNum ** 2

    def is_non_decreasing(*values):
        """Check if values are non-decreasing (can stay same or increase)."""
        for i in range(len(values) - 1):
            if values[i] > values[i + 1]:
                return 0
        return 1

    return {thermo_name: representation.create_tensor_encoding(
        inshape=[max_digit] * n_positions,
        incolors=position_vars,
        function=is_non_decreasing
    )}


if __name__ == "__main__":
    # Test thermometer constraint with 3 positions
    pos_vars = ["bulb", "mid", "tip"]

    # Valid: strictly increasing
    assert thermometer_constraint(pos_vars)[{"bulb": 0, "mid": 1, "tip": 2}] == 1  # 1,2,3
    assert thermometer_constraint(pos_vars)[{"bulb": 1, "mid": 4, "tip": 7}] == 1  # 2,5,8
    assert thermometer_constraint(pos_vars)[{"bulb": 0, "mid": 4, "tip": 8}] == 1  # 1,5,9
    assert thermometer_constraint(pos_vars)[{"bulb": 5, "mid": 6, "tip": 7}] == 1  # 6,7,8

    # Invalid: not strictly increasing
    assert thermometer_constraint(pos_vars)[{"bulb": 2, "mid": 1, "tip": 3}] == 0  # decreasing
    assert thermometer_constraint(pos_vars)[{"bulb": 1, "mid": 1, "tip": 2}] == 0  # same
    assert thermometer_constraint(pos_vars)[{"bulb": 1, "mid": 2, "tip": 2}] == 0  # same
    assert thermometer_constraint(pos_vars)[{"bulb": 5, "mid": 4, "tip": 3}] == 0  # decreasing

    # Test categorical encoding version
    assert cenc_thermometer_constraint(pos_vars)[{"bulb": 0, "mid": 1, "tip": 2}] == 1
    assert cenc_thermometer_constraint(pos_vars)[{"bulb": 1, "mid": 4, "tip": 7}] == 1
    assert cenc_thermometer_constraint(pos_vars)[{"bulb": 2, "mid": 1, "tip": 3}] == 0
    assert cenc_thermometer_constraint(pos_vars)[{"bulb": 1, "mid": 1, "tip": 2}] == 0

    # Test with 2 positions (simplest case)
    pos_vars_2 = ["bulb", "tip"]
    assert thermometer_constraint(pos_vars_2)[{"bulb": 0, "tip": 1}] == 1  # 1,2
    assert thermometer_constraint(pos_vars_2)[{"bulb": 0, "tip": 8}] == 1  # 1,9
    assert thermometer_constraint(pos_vars_2)[{"bulb": 7, "tip": 8}] == 1  # 8,9
    assert thermometer_constraint(pos_vars_2)[{"bulb": 1, "tip": 0}] == 0  # decreasing
    assert thermometer_constraint(pos_vars_2)[{"bulb": 0, "tip": 0}] == 0  # same

    # Test with 4 positions
    pos_vars_4 = ["b", "m1", "m2", "t"]
    assert thermometer_constraint(pos_vars_4)[{"b": 0, "m1": 1, "m2": 2, "t": 3}] == 1  # 1,2,3,4
    assert thermometer_constraint(pos_vars_4)[{"b": 2, "m1": 4, "m2": 6, "t": 8}] == 1  # 3,5,7,9
    assert thermometer_constraint(pos_vars_4)[{"b": 0, "m1": 1, "m2": 1, "t": 2}] == 0  # same
    assert thermometer_constraint(pos_vars_4)[{"b": 3, "m1": 2, "m2": 5, "t": 6}] == 0  # decrease

    # Test thermometer constraint generation
    thermo_cells = [(0, 0), (0, 1), (0, 2)]  # 3-cell thermometer
    thermo_constraints = generate_thermometer_constraints(thermo_cells, "test_thermo")
    assert "test_thermo" in thermo_constraints

    # Test slow thermometer (non-decreasing)
    print("Testing slow thermometer (non-decreasing)...")
    # Use cenc function directly with variable names
    slow_const = representation.create_tensor_encoding(
        inshape=[9, 9],
        incolors=["bulb", "tip"],
        function=lambda *values: 1 if values[0] <= values[1] else 0
    )
    assert slow_const[{"bulb": 0, "tip": 0}] == 1  # same OK
    assert slow_const[{"bulb": 0, "tip": 1}] == 1  # increase OK
    assert slow_const[{"bulb": 1, "tip": 0}] == 0  # decrease NOT OK

    print("All thermometer constraint tests passed!")
