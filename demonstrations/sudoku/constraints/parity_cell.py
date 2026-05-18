from demonstrations.sudoku.constraints._constraint_utils import digit_predicate_core, digit_value


def parity_cell_constraint(posVar, parity, sudokuNum=3, coreType=None):
    if parity not in {"odd", "even", 0, 1}:
        raise ValueError("parity must be 'odd', 'even', 1, or 0")
    expected = 1 if parity in {"odd", 1} else 0
    return digit_predicate_core(
        [posVar],
        lambda val: digit_value(val) % 2 == expected,
        sudokuNum=sudokuNum,
        name=f"{parity}_{posVar}",
        coreType=coreType,
    )


def odd_cell_constraint(posVar, sudokuNum=3, coreType=None):
    return parity_cell_constraint(posVar, "odd", sudokuNum=sudokuNum, coreType=coreType)


def even_cell_constraint(posVar, sudokuNum=3, coreType=None):
    return parity_cell_constraint(posVar, "even", sudokuNum=sudokuNum, coreType=coreType)

