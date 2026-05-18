from demonstrations.sudoku.constraints._constraint_utils import digit_predicate_core, digit_value


def modular_window_constraint(posVar1, posVar2, posVar3, sudokuNum=3, modulus=3, coreType=None):
    expected = list(range(modulus))
    return digit_predicate_core(
        [posVar1, posVar2, posVar3],
        lambda val1, val2, val3: sorted(digit_value(val) % modulus for val in (val1, val2, val3)) == expected,
        sudokuNum=sudokuNum,
        name=f"modular_{posVar1}_{posVar2}_{posVar3}",
        coreType=coreType,
    )


def modular_line_constraints(position_vars, sudokuNum=3, modulus=3, prefix="modular", coreType=None):
    position_vars = list(position_vars)
    return {
        f"{prefix}_{index}": modular_window_constraint(
            position_vars[index],
            position_vars[index + 1],
            position_vars[index + 2],
            sudokuNum=sudokuNum,
            modulus=modulus,
            coreType=coreType,
        )
        for index in range(len(position_vars) - 2)
    }

