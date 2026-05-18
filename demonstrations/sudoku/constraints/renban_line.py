from demonstrations.sudoku.constraints._constraint_utils import digit_predicate_core


def renban_line_constraint(position_vars, sudokuNum=3, coreType=None, max_dense_cells=5_000_000):
    position_vars = list(position_vars)

    def is_renban(*values):
        if len(set(values)) != len(values):
            return False
        return max(values) - min(values) == len(values) - 1

    return digit_predicate_core(
        position_vars,
        is_renban,
        sudokuNum=sudokuNum,
        name="renban_" + "_".join(position_vars),
        coreType=coreType,
        max_dense_cells=max_dense_cells,
    )


def renban_line_constraints(lines, sudokuNum=3, coreType=None):
    return {
        f"renban_{index}": renban_line_constraint(line, sudokuNum=sudokuNum, coreType=coreType)
        for index, line in enumerate(lines)
        if len(line) >= 2
    }

