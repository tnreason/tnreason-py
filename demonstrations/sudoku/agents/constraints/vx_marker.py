"""
V/X Marker Constraints for Sudoku

Natural Language Rules:
- V Marker: "Digits separated by a V sum to 5"
- X Marker: "Digits separated by an X sum to 10"

These constraints apply to pairs of adjacent cells marked with V or X symbols.
Note: Some puzzles state "Not all possible Vs and Xes are given" meaning there
may be other pairs that satisfy the sum condition but are not marked.

V Marker valid pairs (1-indexed): (1,4), (2,3), (3,2), (4,1)
  In 0-indexed: (0,3), (1,2), (2,1), (3,2)

X Marker valid pairs (1-indexed): (1,9), (2,8), (3,7), (4,6), (5,5), (6,4), (7,3), (8,2), (9,1)
  In 0-indexed: (0,8), (1,7), (2,6), (3,5), (4,4), (5,3), (6,2), (7,1), (8,0)
"""

from tnreason import engine
from tnreason import representation


def v_marker_constraint(posVar1, posVar2, sudokuNum=3, target_sum=5):
    """
    Creates a constraint that two positions separated by a V marker
    must have values that sum to 5.

    Args:
        posVar1: Variable name for first position
        posVar2: Variable name for second position
        sudokuNum: Size parameter (default 3 for 9x9 sudoku, giving values 1-9)
        target_sum: Target sum (default 5 for V marker)

    Returns:
        Dictionary with V marker constraint tensor
    """
    # Values are 0-indexed, so we need (v1+1) + (v2+1) = target_sum
    # Which means v1 + v2 = target_sum - 2
    adjusted_sum = target_sum - 2

    return engine.create_from_slice_iterator(
        shape=[sudokuNum ** 2, sudokuNum ** 2],
        colors=[posVar1, posVar2],
        sliceIterator=[(1, {posVar1: val1, posVar2: val2})
                       for val1 in range(sudokuNum ** 2)
                       for val2 in range(sudokuNum ** 2)
                       if val1 + val2 == adjusted_sum]
    )


def x_marker_constraint(posVar1, posVar2, sudokuNum=3, target_sum=10):
    """
    Creates a constraint that two positions separated by an X marker
    must have values that sum to 10.

    Args:
        posVar1: Variable name for first position
        posVar2: Variable name for second position
        sudokuNum: Size parameter (default 3 for 9x9 sudoku, giving values 1-9)
        target_sum: Target sum (default 10 for X marker)

    Returns:
        Dictionary with X marker constraint tensor
    """
    # Values are 0-indexed, so we need (v1+1) + (v2+1) = target_sum
    # Which means v1 + v2 = target_sum - 2
    adjusted_sum = target_sum - 2

    return engine.create_from_slice_iterator(
        shape=[sudokuNum ** 2, sudokuNum ** 2],
        colors=[posVar1, posVar2],
        sliceIterator=[(1, {posVar1: val1, posVar2: val2})
                       for val1 in range(sudokuNum ** 2)
                       for val2 in range(sudokuNum ** 2)
                       if val1 + val2 == adjusted_sum]
    )


def cenc_v_marker_constraint(posVar1, posVar2, sudokuNum=3, target_sum=5):
    """
    Creates a constraint that two positions separated by a V marker
    must have values that sum to 5.
    Uses the representation encoding approach.

    Args:
        posVar1: Variable name for first position
        posVar2: Variable name for second position
        sudokuNum: Size parameter (default 3 for 9x9 sudoku)
        target_sum: Target sum (default 5 for V marker)

    Returns:
        Tensor encoding for V marker constraint
    """
    adjusted_sum = target_sum - 2

    return representation.create_tensor_encoding(
        inshape=[sudokuNum ** 2, sudokuNum ** 2],
        incolors=[posVar1, posVar2],
        function=lambda v1, v2: 1 if v1 + v2 == adjusted_sum else 0
    )


def cenc_x_marker_constraint(posVar1, posVar2, sudokuNum=3, target_sum=10):
    """
    Creates a constraint that two positions separated by an X marker
    must have values that sum to 10.
    Uses the representation encoding approach.

    Args:
        posVar1: Variable name for first position
        posVar2: Variable name for second position
        sudokuNum: Size parameter (default 3 for 9x9 sudoku)
        target_sum: Target sum (default 10 for X marker)

    Returns:
        Tensor encoding for X marker constraint
    """
    adjusted_sum = target_sum - 2

    return representation.create_tensor_encoding(
        inshape=[sudokuNum ** 2, sudokuNum ** 2],
        incolors=[posVar1, posVar2],
        function=lambda v1, v2: 1 if v1 + v2 == adjusted_sum else 0
    )


def generate_vx_constraints(marker_pairs, marker_type="v"):
    """
    Generates V or X marker constraints for pairs of cells.

    Args:
        marker_pairs: List of ((row1, col1), (row2, col2)) tuples defining marker positions
        marker_type: "v" for V markers (sum to 5), "x" for X markers (sum to 10)

    Returns:
        Dictionary of marker constraints
    """
    if marker_type not in ["v", "x"]:
        raise ValueError("marker_type must be 'v' or 'x'")

    target_sum = 5 if marker_type == "v" else 10
    constraint_func = cenc_v_marker_constraint if marker_type == "v" else cenc_x_marker_constraint

    constraints = {}

    for i, ((row1, col1), (row2, col2)) in enumerate(marker_pairs):
        posVar1 = f"r{row1}c{col1}"
        posVar2 = f"r{row2}c{col2}"

        constraint_name = f"{marker_type}_marker_{i}"
        constraints[constraint_name] = constraint_func(posVar1, posVar2, target_sum=target_sum)

    return constraints


if __name__ == "__main__":
    # Test V marker constraint (sum to 5)
    # Valid pairs (1-indexed): (1,4), (2,3), (3,2), (4,1)
    # In 0-indexed: (0,3), (1,2), (2,1), (3,0)
    assert v_marker_constraint("p1", "p2")[{"p1": 0, "p2": 3}] == 1  # 1+4=5
    assert v_marker_constraint("p1", "p2")[{"p1": 1, "p2": 2}] == 1  # 2+3=5
    assert v_marker_constraint("p1", "p2")[{"p1": 2, "p2": 1}] == 1  # 3+2=5
    assert v_marker_constraint("p1", "p2")[{"p1": 3, "p2": 0}] == 1  # 4+1=5

    # Invalid pairs
    assert v_marker_constraint("p1", "p2")[{"p1": 0, "p2": 0}] == 0  # 1+1=2
    assert v_marker_constraint("p1", "p2")[{"p1": 4, "p2": 4}] == 0  # 5+5=10
    assert v_marker_constraint("p1", "p2")[{"p1": 0, "p2": 8}] == 0  # 1+9=10
    assert v_marker_constraint("p1", "p2")[{"p1": 1, "p2": 3}] == 0  # 2+4=6

    # Test X marker constraint (sum to 10)
    # Valid pairs (1-indexed): (1,9), (2,8), (3,7), (4,6), (5,5), (6,4), (7,3), (8,2), (9,1)
    # In 0-indexed: (0,8), (1,7), (2,6), (3,5), (4,4), (5,3), (6,2), (7,1), (8,0)
    assert x_marker_constraint("p1", "p2")[{"p1": 0, "p2": 8}] == 1  # 1+9=10
    assert x_marker_constraint("p1", "p2")[{"p1": 1, "p2": 7}] == 1  # 2+8=10
    assert x_marker_constraint("p1", "p2")[{"p1": 4, "p2": 4}] == 1  # 5+5=10
    assert x_marker_constraint("p1", "p2")[{"p1": 8, "p2": 0}] == 1  # 9+1=10

    # Invalid pairs
    assert x_marker_constraint("p1", "p2")[{"p1": 0, "p2": 0}] == 0  # 1+1=2
    assert x_marker_constraint("p1", "p2")[{"p1": 0, "p2": 3}] == 0  # 1+4=5
    assert x_marker_constraint("p1", "p2")[{"p1": 1, "p2": 2}] == 0  # 2+3=5
    
    # Valid: 4+6=10
    assert x_marker_constraint("p1", "p2")[{"p1": 3, "p2": 5}] == 1  # 4+6=10

    # Test categorical encoding versions
    assert cenc_v_marker_constraint("p1", "p2")[{"p1": 0, "p2": 3}] == 1
    assert cenc_v_marker_constraint("p1", "p2")[{"p1": 1, "p2": 2}] == 1
    assert cenc_v_marker_constraint("p1", "p2")[{"p1": 0, "p2": 0}] == 0

    assert cenc_x_marker_constraint("p1", "p2")[{"p1": 0, "p2": 8}] == 1
    assert cenc_x_marker_constraint("p1", "p2")[{"p1": 4, "p2": 4}] == 1
    assert cenc_x_marker_constraint("p1", "p2")[{"p1": 0, "p2": 0}] == 0

    # Test constraint generation
    v_pairs = [((0, 0), (0, 1)), ((1, 2), (1, 3))]
    v_constraints = generate_vx_constraints(v_pairs, "v")
    assert len(v_constraints) == 2
    assert "v_marker_0" in v_constraints
    assert "v_marker_1" in v_constraints

    x_pairs = [((2, 2), (2, 3))]
    x_constraints = generate_vx_constraints(x_pairs, "x")
    assert len(x_constraints) == 1
    assert "x_marker_0" in x_constraints

    print("All V/X marker constraint tests passed!")
