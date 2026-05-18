from demonstrations.sudoku.constraints._constraint_utils import digit_value, pair_constraint


def red_dot_constraint(posVar1, posVar2, sudokuNum=3, coreType=None):
    """
    Red-dot rule used by challenge100 row 52: the digits are non-consecutive
    and have opposite parity.
    """
    return pair_constraint(
        posVar1,
        posVar2,
        lambda val1, val2: (
            abs(digit_value(val1) - digit_value(val2)) > 1
            and digit_value(val1) % 2 != digit_value(val2) % 2
        ),
        sudokuNum=sudokuNum,
        name=f"red_dot_{posVar1}_{posVar2}",
        coreType=coreType,
    )

