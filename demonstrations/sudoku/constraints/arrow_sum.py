from demonstrations.sudoku.constraints._constraint_utils import digit_predicate_core, digit_value, sum_values


def arrow_sum_constraint(circle_var, arrow_vars, sudokuNum=3, coreType=None, max_dense_cells=5_000_000):
    arrow_vars = list(arrow_vars)
    colors = [circle_var] + arrow_vars

    return digit_predicate_core(
        colors,
        lambda circle, *arrow_values: digit_value(circle) == sum_values(arrow_values),
        sudokuNum=sudokuNum,
        name="arrow_" + circle_var,
        coreType=coreType,
        max_dense_cells=max_dense_cells,
    )


def arrow_sum_constraints(arrows, sudokuNum=3, coreType=None):
    constraints = {}
    for index, arrow in enumerate(arrows):
        if len(arrow) == 3:
            circle_var, arrow_vars, name = arrow
        else:
            circle_var, arrow_vars = arrow
            name = f"arrow_{index}"
        constraints[name] = arrow_sum_constraint(circle_var, arrow_vars, sudokuNum=sudokuNum, coreType=coreType)
    return constraints

