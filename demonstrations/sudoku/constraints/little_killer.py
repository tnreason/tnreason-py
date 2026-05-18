from demonstrations.sudoku.constraints._constraint_utils import digit_predicate_core, sum_to_hidden_core, sum_values


def little_killer_constraint(position_vars, target_sum, sudokuNum=3, coreType=None, max_dense_cells=5_000_000):
    position_vars = list(position_vars)
    return digit_predicate_core(
        position_vars,
        lambda *values: sum_values(values) == target_sum,
        sudokuNum=sudokuNum,
        name="little_killer_" + "_".join(position_vars),
        coreType=coreType,
        max_dense_cells=max_dense_cells,
    )


def little_killer_to_hidden_constraint(
    position_vars,
    sum_var,
    min_sum,
    max_sum,
    sudokuNum=3,
    coreType=None,
    max_dense_cells=5_000_000,
):
    return sum_to_hidden_core(
        sum_var=sum_var,
        position_vars=position_vars,
        min_sum=min_sum,
        max_sum=max_sum,
        sudokuNum=sudokuNum,
        name="little_killer_hidden_" + "_".join(position_vars),
        coreType=coreType,
        max_dense_cells=max_dense_cells,
    )

