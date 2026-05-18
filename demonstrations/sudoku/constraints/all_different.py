from demonstrations.sudoku.constraints._constraint_utils import digit_predicate_core
from demonstrations.sudoku.constraints.anti_knight import anti_knight_constraint


def all_different_constraint(
    position_vars, sudokuNum=3, coreType=None, max_dense_cells=5_000_000
):
    position_vars = list(position_vars)
    return digit_predicate_core(
        position_vars,
        lambda *values: len(set(values)) == len(values),
        sudokuNum=sudokuNum,
        name="all_different_" + "_".join(position_vars),
        coreType=coreType,
        max_dense_cells=max_dense_cells,
    )


def all_different_pair_constraints(
    position_vars, sudokuNum=3, prefix="all_different", coreType=None
):
    position_vars = list(position_vars)
    constraints = {}
    for left_index, left in enumerate(position_vars):
        for right in position_vars[left_index + 1 :]:
            constraints[f"{prefix}_{left}_{right}"] = anti_knight_constraint(
                left,
                right,
                sudokuNum=sudokuNum,
                coreType=coreType,
            )
    return constraints
