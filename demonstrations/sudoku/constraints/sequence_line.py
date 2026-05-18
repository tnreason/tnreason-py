from demonstrations.sudoku.constraints._constraint_utils import digit_predicate_core, digit_value


def sequence_line_constraint(
    position_vars,
    sudokuNum=3,
    increasing=True,
    allow_zero_difference=False,
    coreType=None,
    max_dense_cells=5_000_000,
):
    """
    Arithmetic sequence line: adjacent differences are constant in line order.
    """
    position_vars = list(position_vars)

    def is_valid(*values):
        digits = [digit_value(value) for value in values]
        if len(digits) < 2:
            return True
        difference = digits[1] - digits[0]
        if increasing and difference < 0:
            return False
        if not allow_zero_difference and difference == 0:
            return False
        return all(digits[index + 1] - digits[index] == difference for index in range(len(digits) - 1))

    return digit_predicate_core(
        position_vars,
        is_valid,
        sudokuNum=sudokuNum,
        name="sequence_" + "_".join(position_vars),
        coreType=coreType,
        max_dense_cells=max_dense_cells,
    )

