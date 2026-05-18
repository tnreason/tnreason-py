from demonstrations.sudoku.constraints._constraint_utils import pair_constraint


KING_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def anti_king_constraint(posVar1, posVar2, sudokuNum=3, coreType=None):
    return pair_constraint(
        posVar1,
        posVar2,
        lambda val1, val2: val1 != val2,
        sudokuNum=sudokuNum,
        name=f"anti_king_{posVar1}_{posVar2}",
        coreType=coreType,
    )


def king_neighbors(row, col, row_count=9, col_count=9):
    neighbors = []
    for row_delta, col_delta in KING_OFFSETS:
        next_row = row + row_delta
        next_col = col + col_delta
        if 0 <= next_row < row_count and 0 <= next_col < col_count:
            neighbors.append((next_row, next_col))
    return neighbors


def all_anti_king_constraints(row_count=9, col_count=9, cell_var=None, sudokuNum=3, coreType=None):
    cell_var = cell_var or (lambda row, col: f"r{row}c{col}")
    constraints = {}
    for row in range(row_count):
        for col in range(col_count):
            for next_row, next_col in king_neighbors(row, col, row_count=row_count, col_count=col_count):
                if (row, col) < (next_row, next_col):
                    posVar1 = cell_var(row, col)
                    posVar2 = cell_var(next_row, next_col)
                    constraints[f"anti_king_{posVar1}_{posVar2}"] = anti_king_constraint(
                        posVar1,
                        posVar2,
                        sudokuNum=sudokuNum,
                        coreType=coreType,
                    )
    return constraints

