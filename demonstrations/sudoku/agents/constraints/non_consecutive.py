"""
Non-Consecutive Constraint for Sudoku (Red Dot Variant)

Natural Language Rule: "Digits separated by a red dot are non-consecutive and 
                       contain one even digit and one odd digit."
                       "All dots are given!" (meaning: if a red dot could be placed, it is)

A red dot between two cells requires:
1. The digits are NOT consecutive (|v1 - v2| > 1)
2. One digit is even and one is odd (different parity)

This is different from the white dot (consecutive) and black dot (1:2 ratio).

Examples of valid red dot pairs:
- [1, 4] - not consecutive (gap=3), 1 is odd, 4 is even
- [2, 5] - not consecutive (gap=3), 2 is even, 5 is odd
- [3, 8] - not consecutive (gap=5), 3 is odd, 8 is even
- [4, 7] - not consecutive (gap=3), 4 is even, 7 is odd

Examples of invalid red dot pairs:
- [1, 2] - consecutive (forbidden)
- [2, 4] - same parity (both even)
- [1, 3] - same parity (both odd)
- [5, 5] - same value (also same parity)
"""

from tnreason import engine
from tnreason import representation


def non_consecutive_constraint(posVar1, posVar2, sudokuNum=3):
    """
    Creates a constraint that two positions separated by a red dot must contain
    non-consecutive digits with different parity (one even, one odd).

    Args:
        posVar1: Variable name for first position
        posVar2: Variable name for second position
        sudokuNum: Size parameter (default 3 for 9x9 sudoku, giving values 1-9)

    Returns:
        Dictionary with non-consecutive constraint tensor
    """
    return engine.create_from_slice_iterator(
        shape=[sudokuNum ** 2, sudokuNum ** 2],
        colors=[posVar1, posVar2],
        sliceIterator=[(1, {posVar1: val1, posVar2: val2})
                       for val1 in range(sudokuNum ** 2)
                       for val2 in range(sudokuNum ** 2)
                       if abs(val1 - val2) > 1 and (val1 % 2) != (val2 % 2)]
    )


def cenc_non_consecutive_constraint(posVar1, posVar2, sudokuNum=3):
    """
    Creates a constraint that two positions separated by a red dot must contain
    non-consecutive digits with different parity (one even, one odd).
    Uses the representation encoding approach.

    Args:
        posVar1: Variable name for first position
        posVar2: Variable name for second position
        sudokuNum: Size parameter (default 3 for 9x9 sudoku)

    Returns:
        Tensor encoding for non-consecutive constraint
    """
    return representation.create_tensor_encoding(
        inshape=[sudokuNum ** 2, sudokuNum ** 2],
        incolors=[posVar1, posVar2],
        function=lambda v1, v2: 1 if (abs(v1 - v2) > 1 and (v1 % 2) != (v2 % 2)) else 0
    )


def generate_non_consecutive_constraints(marker_pairs, marker_name="non_consecutive"):
    """
    Generates non-consecutive (red dot) constraints for pairs of cells.

    Args:
        marker_pairs: List of ((row1, col1), (row2, col2)) tuples defining red dot positions
        marker_name: Base name for the constraints

    Returns:
        Dictionary of non-consecutive constraints
    """
    constraints = {}

    for i, ((row1, col1), (row2, col2)) in enumerate(marker_pairs):
        posVar1 = f"r{row1}c{col1}"
        posVar2 = f"r{row2}c{col2}"

        constraint_name = f"{marker_name}_{i}"
        constraints[constraint_name] = cenc_non_consecutive_constraint(posVar1, posVar2)

    return constraints


if __name__ == "__main__":
    # Test non-consecutive constraint
    # Valid: non-consecutive AND different parity
    print("Testing valid pairs...")
    assert non_consecutive_constraint("p1", "p2")[{"p1": 0, "p2": 3}] == 1  # 1,4 (odd,even)
    assert non_consecutive_constraint("p1", "p2")[{"p1": 3, "p2": 0}] == 1  # 4,1 (even,odd)
    assert non_consecutive_constraint("p1", "p2")[{"p1": 1, "p2": 4}] == 1  # 2,5 (even,odd)
    assert non_consecutive_constraint("p1", "p2")[{"p1": 2, "p2": 7}] == 1  # 3,8 (odd,even)
    assert non_consecutive_constraint("p1", "p2")[{"p1": 0, "p2": 5}] == 1  # 1,6 (odd,even)

    # Invalid: consecutive (regardless of parity)
    print("Testing consecutive pairs (should be invalid)...")
    assert non_consecutive_constraint("p1", "p2")[{"p1": 0, "p2": 1}] == 0  # 1,2 consecutive
    assert non_consecutive_constraint("p1", "p2")[{"p1": 1, "p2": 2}] == 0  # 2,3 consecutive
    assert non_consecutive_constraint("p1", "p2")[{"p1": 2, "p2": 3}] == 0  # 3,4 consecutive
    assert non_consecutive_constraint("p1", "p2")[{"p1": 7, "p2": 8}] == 0  # 8,9 consecutive

    # Invalid: same parity (regardless of gap)
    print("Testing same parity pairs (should be invalid)...")
    assert non_consecutive_constraint("p1", "p2")[{"p1": 0, "p2": 2}] == 0  # 1,3 both odd
    assert non_consecutive_constraint("p1", "p2")[{"p1": 0, "p2": 4}] == 0  # 1,5 both odd
    assert non_consecutive_constraint("p1", "p2")[{"p1": 0, "p2": 8}] == 0  # 1,9 both odd
    assert non_consecutive_constraint("p1", "p2")[{"p1": 1, "p2": 3}] == 0  # 2,4 both even
    assert non_consecutive_constraint("p1", "p2")[{"p1": 1, "p2": 5}] == 0  # 2,6 both even

    # Invalid: same value
    print("Testing same value pairs (should be invalid)...")
    assert non_consecutive_constraint("p1", "p2")[{"p1": 0, "p2": 0}] == 0  # 1,1
    assert non_consecutive_constraint("p1", "p2")[{"p1": 4, "p2": 4}] == 0  # 5,5

    # Test categorical encoding version
    print("Testing categorical encoding version...")
    assert cenc_non_consecutive_constraint("p1", "p2")[{"p1": 0, "p2": 3}] == 1
    assert cenc_non_consecutive_constraint("p1", "p2")[{"p1": 1, "p2": 4}] == 1
    assert cenc_non_consecutive_constraint("p1", "p2")[{"p1": 0, "p2": 1}] == 0
    assert cenc_non_consecutive_constraint("p1", "p2")[{"p1": 0, "p2": 2}] == 0

    # Test constraint generation
    print("Testing constraint generation...")
    marker_pairs = [((0, 0), (0, 1)), ((1, 2), (1, 3))]
    constraints = generate_non_consecutive_constraints(marker_pairs, "test_red")
    assert len(constraints) == 2
    assert "test_red_0" in constraints
    assert "test_red_1" in constraints

    print("All non-consecutive constraint tests passed!")
