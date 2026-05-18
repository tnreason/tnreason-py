from demonstrations.sudoku.constraints._constraint_utils import digit_predicate_core, digit_value


def balanced_four_cell_line_constraint(position_vars, sudokuNum=3, coreType=None):
    position_vars = list(position_vars)
    if len(position_vars) != 4:
        raise ValueError("balanced_four_cell_line_constraint expects exactly four position variables")
    return digit_predicate_core(
        position_vars,
        lambda val1, val2, val3, val4: digit_value(val1) + digit_value(val2) == digit_value(val3) + digit_value(val4),
        sudokuNum=sudokuNum,
        name="balanced_" + "_".join(position_vars),
        coreType=coreType,
    )

