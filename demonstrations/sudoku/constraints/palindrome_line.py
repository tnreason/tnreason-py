from demonstrations.sudoku.constraints._constraint_utils import digit_predicate_core, pair_constraint


def equal_pair_constraint(posVar1, posVar2, sudokuNum=3, coreType=None):
    return pair_constraint(
        posVar1,
        posVar2,
        lambda val1, val2: val1 == val2,
        sudokuNum=sudokuNum,
        name=f"equal_{posVar1}_{posVar2}",
        coreType=coreType,
    )


def palindrome_line_constraint(position_vars, sudokuNum=3, coreType=None, max_dense_cells=5_000_000):
    position_vars = list(position_vars)
    return digit_predicate_core(
        position_vars,
        lambda *values: all(values[index] == values[-index - 1] for index in range(len(values) // 2)),
        sudokuNum=sudokuNum,
        name="palindrome_" + "_".join(position_vars),
        coreType=coreType,
        max_dense_cells=max_dense_cells,
    )


def palindrome_pair_constraints(position_vars, sudokuNum=3, prefix="palindrome", coreType=None):
    position_vars = list(position_vars)
    constraints = {}
    for index in range(len(position_vars) // 2):
        posVar1 = position_vars[index]
        posVar2 = position_vars[-index - 1]
        constraints[f"{prefix}_{index}"] = equal_pair_constraint(
            posVar1,
            posVar2,
            sudokuNum=sudokuNum,
            coreType=coreType,
        )
    return constraints

