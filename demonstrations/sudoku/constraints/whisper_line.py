from demonstrations.sudoku.constraints._constraint_utils import digit_value, line_adjacent_constraints, pair_constraint


def whisper_pair_constraint(posVar1, posVar2, min_difference=5, sudokuNum=3, coreType=None):
    return pair_constraint(
        posVar1,
        posVar2,
        lambda val1, val2: abs(digit_value(val1) - digit_value(val2)) >= min_difference,
        sudokuNum=sudokuNum,
        name=f"whisper_{posVar1}_{posVar2}",
        coreType=coreType,
    )


def german_whisper_constraint(posVar1, posVar2, sudokuNum=3, min_difference=5, coreType=None):
    return whisper_pair_constraint(
        posVar1,
        posVar2,
        min_difference=min_difference,
        sudokuNum=sudokuNum,
        coreType=coreType,
    )


def dutch_whisper_constraint(posVar1, posVar2, sudokuNum=3, coreType=None):
    return whisper_pair_constraint(posVar1, posVar2, min_difference=4, sudokuNum=sudokuNum, coreType=coreType)


def whisper_line_constraints(position_vars, min_difference=5, sudokuNum=3, prefix="whisper", coreType=None):
    return line_adjacent_constraints(
        list(position_vars),
        lambda posVar1, posVar2: whisper_pair_constraint(
            posVar1,
            posVar2,
            min_difference=min_difference,
            sudokuNum=sudokuNum,
            coreType=coreType,
        ),
        prefix,
    )

