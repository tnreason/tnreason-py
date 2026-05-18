from demonstrations.sudoku.constraints._constraint_utils import digit_predicate_core, digit_value


def quadruple_constraint(position_vars, required_digits, sudokuNum=3, coreType=None, max_dense_cells=5_000_000):
    position_vars = list(position_vars)
    required_digits = set(int(digit) for digit in required_digits)

    return digit_predicate_core(
        position_vars,
        lambda *values: required_digits <= {digit_value(value) for value in values},
        sudokuNum=sudokuNum,
        name="quadruple_" + "_".join(position_vars),
        coreType=coreType,
        max_dense_cells=max_dense_cells,
    )

