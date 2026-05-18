from demonstrations.sudoku.constraints._constraint_utils import digit_predicate_core, digit_value


def lockout_line_constraint(position_vars, min_endpoint_difference=4, sudokuNum=3, coreType=None, max_dense_cells=5_000_000):
    """
    Lockout line: endpoints differ by at least ``min_endpoint_difference`` and
    every interior digit lies outside the interval between the endpoints.
    """
    position_vars = list(position_vars)
    if len(position_vars) < 3:
        raise ValueError("lockout_line_constraint expects at least three position variables")

    def is_valid(*values):
        digits = [digit_value(value) for value in values]
        low = min(digits[0], digits[-1])
        high = max(digits[0], digits[-1])
        if high - low < min_endpoint_difference:
            return False
        return all(digit < low or digit > high for digit in digits[1:-1])

    return digit_predicate_core(
        position_vars,
        is_valid,
        sudokuNum=sudokuNum,
        name="lockout_" + "_".join(position_vars),
        coreType=coreType,
        max_dense_cells=max_dense_cells,
    )

