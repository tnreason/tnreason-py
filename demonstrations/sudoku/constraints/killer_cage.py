from demonstrations.sudoku.constraints._constraint_utils import digit_predicate_core, sum_values


def killer_cage_constraint(
    position_vars,
    target_sum,
    sudokuNum=3,
    allow_repeats=False,
    coreType=None,
    max_dense_cells=5_000_000,
):
    position_vars = list(position_vars)

    def is_valid(*values):
        if not allow_repeats and len(set(values)) != len(values):
            return False
        return sum_values(values) == target_sum

    return digit_predicate_core(
        position_vars,
        is_valid,
        sudokuNum=sudokuNum,
        name="killer_" + "_".join(position_vars),
        coreType=coreType,
        max_dense_cells=max_dense_cells,
    )


def killer_cage_constraints(cages, sudokuNum=3, coreType=None):
    constraints = {}
    for index, cage in enumerate(cages):
        if len(cage) == 3:
            position_vars, target_sum, name = cage
        else:
            position_vars, target_sum = cage
            name = f"killer_cage_{index}"
        constraints[name] = killer_cage_constraint(
            position_vars,
            target_sum,
            sudokuNum=sudokuNum,
            coreType=coreType,
        )
    return constraints

