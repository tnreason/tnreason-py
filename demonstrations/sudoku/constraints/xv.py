from demonstrations.sudoku.constraints._constraint_utils import digit_value, pair_constraint


def sum_pair_constraint(posVar1, posVar2, target_sum, sudokuNum=3, coreType=None):
    return pair_constraint(
        posVar1,
        posVar2,
        lambda val1, val2: digit_value(val1) + digit_value(val2) == target_sum,
        sudokuNum=sudokuNum,
        name=f"sum_{target_sum}_{posVar1}_{posVar2}",
        coreType=coreType,
    )


def v_constraint(posVar1, posVar2, sudokuNum=3, coreType=None):
    return sum_pair_constraint(posVar1, posVar2, target_sum=5, sudokuNum=sudokuNum, coreType=coreType)


def x_constraint(posVar1, posVar2, sudokuNum=3, coreType=None):
    return sum_pair_constraint(posVar1, posVar2, target_sum=10, sudokuNum=sudokuNum, coreType=coreType)


def xv_constraints(marker_pairs, marker_type, sudokuNum=3, prefix=None, coreType=None):
    if marker_type.lower() not in {"v", "x"}:
        raise ValueError("marker_type must be 'v' or 'x'")
    factory = v_constraint if marker_type.lower() == "v" else x_constraint
    prefix = prefix or f"{marker_type.lower()}_marker"
    return {
        f"{prefix}_{index}": factory(posVar1, posVar2, sudokuNum=sudokuNum, coreType=coreType)
        for index, (posVar1, posVar2) in enumerate(marker_pairs)
    }


v_marker_constraint = v_constraint
x_marker_constraint = x_constraint

