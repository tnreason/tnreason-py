from demonstrations.sudoku.constraints._constraint_utils import digit_value, line_adjacent_constraints, pair_constraint


def thermometer_pair_constraint(posVar1, posVar2, sudokuNum=3, strict=True, coreType=None):
    if strict:
        predicate = lambda val1, val2: digit_value(val1) < digit_value(val2)
    else:
        predicate = lambda val1, val2: digit_value(val1) <= digit_value(val2)
    return pair_constraint(
        posVar1,
        posVar2,
        predicate,
        sudokuNum=sudokuNum,
        name=f"thermo_{posVar1}_{posVar2}",
        coreType=coreType,
    )


def thermometer_line_constraints(position_vars, sudokuNum=3, strict=True, prefix="thermo", coreType=None):
    return line_adjacent_constraints(
        list(position_vars),
        lambda posVar1, posVar2: thermometer_pair_constraint(
            posVar1,
            posVar2,
            sudokuNum=sudokuNum,
            strict=strict,
            coreType=coreType,
        ),
        prefix,
    )


def slow_thermometer_line_constraints(position_vars, sudokuNum=3, prefix="slow_thermo", coreType=None):
    return thermometer_line_constraints(
        position_vars,
        sudokuNum=sudokuNum,
        strict=False,
        prefix=prefix,
        coreType=coreType,
    )

