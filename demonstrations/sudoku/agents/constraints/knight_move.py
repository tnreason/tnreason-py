"""
Anti-Knight Constraint for Sudoku

Natural Language Rule: "Digits that are a knight's move apart cannot be identical anywhere in the grid."
                       "Cells separated by a knight's move (in chess) cannot contain the same digit."

A knight's move in chess is an L-shape: 2 cells in one direction and 1 cell perpendicular.
For a cell at position (row, col), the knight moves are:
- (row-2, col-1), (row-2, col+1)
- (row-1, col-2), (row-1, col+2)
- (row+1, col-2), (row+1, col+2)
- (row+2, col-1), (row+2, col+1)
"""

from tnreason import engine
from tnreason import representation
from constraints.rc_to_variable_helper import rc_to_pos_assignment


def knight_move_constraint(posVar1, posVar2, sudokuNum=3):
    """
    Creates a constraint that two positions connected by a knight's move cannot have the same value.
    
    Args:
        posVar1: Variable name for first position (e.g., "r0c0")
        posVar2: Variable name for second position (e.g., "r2c1")
        sudokuNum: Size parameter (default 3 for 9x9 sudoku, giving values 1-9)
    
    Returns:
        Dictionary with knight move constraint tensor
    """
    return engine.create_from_slice_iterator(
        shape=[sudokuNum ** 2, sudokuNum ** 2],
        colors=[posVar1, posVar2],
        sliceIterator=[(1, {posVar1: val1, posVar2: val2}) 
                       for val1 in range(sudokuNum ** 2) 
                       for val2 in range(sudokuNum ** 2) 
                       if val1 != val2]
    )


def cenc_knight_move_constraint(posVar1, posVar2, sudokuNum=3):
    """
    Creates a constraint that two positions connected by a knight's move cannot have the same value.
    Uses the representation encoding approach.
    
    Args:
        posVar1: Variable name for first position
        posVar2: Variable name for second position
        sudokuNum: Size parameter (default 3 for 9x9 sudoku)
    
    Returns:
        Tensor encoding for knight move constraint
    """
    return representation.create_tensor_encoding(
        inshape=[sudokuNum ** 2, sudokuNum ** 2], 
        incolors=[posVar1, posVar2],
        function=lambda v1, v2: 1 if v1 != v2 else 0
    )


def get_knight_moves_for_cell(row, col, grid_size=9):
    """
    Returns all valid knight move positions from a given cell.
    
    Args:
        row: Row index (0-based)
        col: Column index (0-based)
        grid_size: Size of the grid (default 9)
    
    Returns:
        List of (row, col) tuples representing valid knight moves
    """
    knight_offsets = [
        (-2, -1), (-2, 1),
        (-1, -2), (-1, 2),
        (1, -2), (1, 2),
        (2, -1), (2, 1)
    ]
    
    valid_moves = []
    for dr, dc in knight_offsets:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < grid_size and 0 <= new_col < grid_size:
            valid_moves.append((new_row, new_col))
    
    return valid_moves


def generate_all_knight_constraints(grid_size=9, sudokuNum=3):
    """
    Generates all knight move constraints for a sudoku grid.
    
    Args:
        grid_size: Size of the grid (default 9)
        sudokuNum: Size parameter (default 3 for 9x9 sudoku)
    
    Returns:
        Dictionary of all knight move constraints
    """
    constraints = {}
    
    # For each cell, generate constraints with all its knight move neighbors
    for row in range(grid_size):
        for col in range(grid_size):
            # posVar1 = f"r{row}c{col}"
            # posVar1 = "pos_" + str(row//sudokuNum) + "_" + str(row%sudokuNum) + "_" + str(col//sudokuNum) + "_" + str(col%sudokuNum)
            posVar1 = rc_to_pos_assignment(row, col)
            knight_moves = get_knight_moves_for_cell(row, col, grid_size)
            
            for km_row, km_col in knight_moves:
                # posVar2 = f"r{km_row}c{km_col}"
                # posVar2 = "pos_" + str(km_row//sudokuNum) + "_" + str(km_row%sudokuNum) + "_" + str(km_col//sudokuNum) + "_" + str(km_col%sudokuNum)
                posVar2 = rc_to_pos_assignment(km_row, km_col)
                # Only create constraint once per pair (avoid duplicates)
                if (row, col) < (km_row, km_col):
                    constraint_name = f"knight_{posVar1}_{posVar2}"
                    constraints[constraint_name] = cenc_knight_move_constraint(posVar1, posVar2, sudokuNum)
    
    return constraints


if __name__ == "__main__":
    # Test knight move constraint
    # Values must be different
    assert knight_move_constraint("p1", "p2")[{"p1": 5, "p2": 6}] == 1
    assert knight_move_constraint("p1", "p2")[{"p1": 5, "p2": 5}] == 0
    assert knight_move_constraint("p1", "p2")[{"p1": 0, "p2": 8}] == 1
    assert knight_move_constraint("p1", "p2")[{"p1": 0, "p2": 0}] == 0
    
    # Test categorical encoding version
    assert cenc_knight_move_constraint("p1", "p2")[{"p1": 5, "p2": 6}] == 1
    assert cenc_knight_move_constraint("p1", "p2")[{"p1": 5, "p2": 5}] == 0
    assert cenc_knight_move_constraint("p1", "p2")[{"p1": 0, "p2": 8}] == 1
    assert cenc_knight_move_constraint("p1", "p2")[{"p1": 0, "p2": 0}] == 0
    
    # Test knight move generation
    # Center cell (4,4) should have 8 knight moves
    center_moves = get_knight_moves_for_cell(4, 4)
    assert len(center_moves) == 8
    assert (2, 3) in center_moves
    assert (6, 5) in center_moves
    
    # Corner cell (0,0) should have 2 knight moves
    corner_moves = get_knight_moves_for_cell(0, 0)
    assert len(corner_moves) == 2
    assert (1, 2) in corner_moves
    assert (2, 1) in corner_moves
    
    # Edge cell (0,4) should have 4 knight moves
    edge_moves = get_knight_moves_for_cell(0, 4)
    assert len(edge_moves) == 4
    
    print("All knight move constraint tests passed!")
