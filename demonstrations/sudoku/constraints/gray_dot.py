from demonstrations.sudoku.constraints._constraint_utils import digit_value, pair_constraint


def gray_dot_constraint(posVar1, posVar2, sudokuNum=3, coreType=None):
    """
    Gray-dot rule used by some variant puzzles: the pair is either consecutive
    or in a 1:2 ratio.
    """
    return pair_constraint(
        posVar1,
        posVar2,
        lambda val1, val2: (
            abs(digit_value(val1) - digit_value(val2)) == 1
            or digit_value(val1) == 2 * digit_value(val2)
            or digit_value(val2) == 2 * digit_value(val1)
        ),
        sudokuNum=sudokuNum,
        name=f"gray_dot_{posVar1}_{posVar2}",
        coreType=coreType,
    )

