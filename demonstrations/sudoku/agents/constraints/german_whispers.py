"""
German Whispers Constraint for Sudoku

Natural Language Rule: "Adjacent digits along a green line must differ by at least 5."
                       "Adjacent digits along a German whisper line must have a difference of at least 5."

This means that for any two adjacent cells on a German whisper line, 
the absolute difference between their values must be >= 5.
Valid pairs include: (1,6), (1,7), (1,8), (1,9), (2,7), (2,8), (2,9), (3,8), (3,9), (4,9)
and their reverses.
"""

from tnreason import engine
from tnreason import representation
from constraints.rc_to_variable_helper import rc_to_pos_assignment


def german_whispers_constraint(posVar1, posVar2, sudokuNum=3):
    """
    Creates a constraint that two adjacent positions on a German whisper line 
    must have values differing by at least 5.
    
    Args:
        posVar1: Variable name for first position
        posVar2: Variable name for second position
        sudokuNum: Size parameter (default 3 for 9x9 sudoku, giving values 1-9)
    
    Returns:
        Dictionary with German whispers constraint tensor
    """
    return engine.create_from_slice_iterator(
        shape=[sudokuNum ** 2, sudokuNum ** 2],
        colors=[posVar1, posVar2],
        sliceIterator=[(1, {posVar1: val1, posVar2: val2}) 
                       for val1 in range(sudokuNum ** 2) 
                       for val2 in range(sudokuNum ** 2) 
                       if abs((val1 + 1) - (val2 + 1)) >= 5]
    )


def cenc_german_whispers_constraint(posVar1, posVar2, sudokuNum=3):
    """
    Creates a constraint that two adjacent positions on a German whisper line
    must have values differing by at least 5.
    Uses the representation encoding approach.
    
    Args:
        posVar1: Variable name for first position
        posVar2: Variable name for second position
        sudokuNum: Size parameter (default 3 for 9x9 sudoku)
    
    Returns:
        Tensor encoding for German whispers constraint
    """
    return representation.create_tensor_encoding(
        inshape=[sudokuNum ** 2, sudokuNum ** 2], 
        incolors=[posVar1, posVar2],
        function=lambda v1, v2: 1 if abs((v1 + 1) - (v2 + 1)) >= 5 else 0
    )


def generate_german_whispers_constraints(line_cells, line_name="german_whisper"):
    """
    Generates German whisper constraints for a line of cells.
    
    Args:
        line_cells: List of (row, col) tuples defining the German whisper line
        line_name: Base name for the constraints
    
    Returns:
        Dictionary of German whisper constraints for adjacent pairs
    """
    if len(line_cells) < 2:
        return {}
    
    constraints = {}
    
    # Create constraints for each adjacent pair
    for i in range(len(line_cells) - 1):
        row1, col1 = line_cells[i]
        row2, col2 = line_cells[i + 1]
        
        # posVar1 = f"r{row1}c{col1}"
        # posVar2 = f"r{row2}c{col2}"
        posVar1 = rc_to_pos_assignment(row1, col1)
        posVar2 = rc_to_pos_assignment(row2, col2)
        
        constraint_name = f"{line_name}_adj_{i}"
        constraints[constraint_name] = cenc_german_whispers_constraint(posVar1, posVar2)
    
    return constraints


if __name__ == "__main__":
    # Test German whispers constraint
    # Valid pairs (difference >= 5)
    assert german_whispers_constraint("p1", "p2")[{"p1": 0, "p2": 5}] == 1  # 1 and 6
    assert german_whispers_constraint("p1", "p2")[{"p1": 0, "p2": 8}] == 1  # 1 and 9
    assert german_whispers_constraint("p1", "p2")[{"p1": 3, "p2": 8}] == 1  # 4 and 9
    assert german_whispers_constraint("p1", "p2")[{"p1": 8, "p2": 0}] == 1  # 9 and 1
    
    # Invalid pairs (difference < 5)
    assert german_whispers_constraint("p1", "p2")[{"p1": 0, "p2": 4}] == 0  # 1 and 5
    assert german_whispers_constraint("p1", "p2")[{"p1": 3, "p2": 7}] == 0  # 4 and 8
    assert german_whispers_constraint("p1", "p2")[{"p1": 4, "p2": 5}] == 0  # 5 and 6
    assert german_whispers_constraint("p1", "p2")[{"p1": 0, "p2": 0}] == 0  # same value
    
    # Test categorical encoding version
    assert cenc_german_whispers_constraint("p1", "p2")[{"p1": 0, "p2": 5}] == 1
    assert cenc_german_whispers_constraint("p1", "p2")[{"p1": 0, "p2": 8}] == 1
    assert cenc_german_whispers_constraint("p1", "p2")[{"p1": 0, "p2": 4}] == 0
    assert cenc_german_whispers_constraint("p1", "p2")[{"p1": 4, "p2": 5}] == 0
    
    # Test line constraint generation
    line = [(0, 0), (0, 1), (0, 2)]  # 3-cell line
    line_constraints = generate_german_whispers_constraints(line, "test_line")
    assert len(line_constraints) == 2  # 2 adjacent pairs
    
    print("All German whispers constraint tests passed!")
