from demonstrations.sudoku.constraints._constraint_utils import digit_value, pair_constraint


def less_than_constraint(smaller_var, larger_var, sudokuNum=3, coreType=None):
    return pair_constraint(
        smaller_var,
        larger_var,
        lambda smaller, larger: digit_value(smaller) < digit_value(larger),
        sudokuNum=sudokuNum,
        name=f"lt_{smaller_var}_{larger_var}",
        coreType=coreType,
    )


def greater_than_constraint(larger_var, smaller_var, sudokuNum=3, coreType=None):
    return less_than_constraint(smaller_var, larger_var, sudokuNum=sudokuNum, coreType=coreType)


def inequality_constraint(posVar1, posVar2, relation="<", sudokuNum=3, coreType=None):
    if relation == "<":
        return less_than_constraint(posVar1, posVar2, sudokuNum=sudokuNum, coreType=coreType)
    if relation == ">":
        return greater_than_constraint(posVar1, posVar2, sudokuNum=sudokuNum, coreType=coreType)
    raise ValueError("relation must be '<' or '>'")

