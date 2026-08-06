from demonstrations.sudoku.constraints._constraint_utils import digit_predicate_core, digit_value


def fibonacci_line_constraint(position_vars, sudokuNum=3, coreType=None, max_dense_cells=5_000_000):
    position_vars = list(position_vars)

    def is_valid(*values):
        for ordered_values in (values, tuple(reversed(values))):
            digits = [digit_value(value) for value in ordered_values]
            if len(digits) < 2:
                return True
            if digits[0] >= digits[1]:
                continue
            if all(
                digits[index] == digits[index - 1] + digits[index - 2]
                for index in range(2, len(digits))
            ):
                return True
        return False

    return digit_predicate_core(
        position_vars,
        is_valid,
        sudokuNum=sudokuNum,
        name="fibonacci_" + "_".join(position_vars),
        coreType=coreType,
        max_dense_cells=max_dense_cells,
    )
