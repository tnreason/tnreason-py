from demonstrations.sudoku.constraints._constraint_utils import digit_count, digit_predicate_core, digit_value, sum_to_hidden_core


def pair_sum_to_digit_constraint(sum_digit_var, posVar1, posVar2, sudokuNum=3, coreType=None):
    return digit_predicate_core(
        [sum_digit_var, posVar1, posVar2],
        lambda total, val1, val2: digit_value(total) == digit_value(val1) + digit_value(val2),
        sudokuNum=sudokuNum,
        name=f"zipper_pair_{posVar1}_{posVar2}",
        coreType=coreType,
    )


def zipper_line_constraints(
    position_vars,
    total_var=None,
    center_is_total=True,
    sudokuNum=3,
    prefix="zipper",
    coreType=None,
):
    """
    Encode zipper pairs: cells equidistant from the center have the same sum.

    If ``center_is_total`` and the line has odd length, the center digit is
    used as the pair sum. Otherwise a hidden ``total_var`` is introduced with
    domain 2..2N.
    """
    position_vars = list(position_vars)
    pair_count = len(position_vars) // 2
    if pair_count == 0:
        return {}

    constraints = {}
    if center_is_total and len(position_vars) % 2 == 1:
        sum_digit_var = position_vars[len(position_vars) // 2]
        for index in range(pair_count):
            constraints[f"{prefix}_{index}"] = pair_sum_to_digit_constraint(
                sum_digit_var,
                position_vars[index],
                position_vars[-index - 1],
                sudokuNum=sudokuNum,
                coreType=coreType,
            )
        return constraints

    total_var = total_var or f"{prefix}_sum"
    max_pair_sum = 2 * digit_count(sudokuNum)
    for index in range(pair_count):
        constraints[f"{prefix}_{index}"] = sum_to_hidden_core(
            sum_var=total_var,
            position_vars=[position_vars[index], position_vars[-index - 1]],
            min_sum=2,
            max_sum=max_pair_sum,
            sudokuNum=sudokuNum,
            name=f"{prefix}_{index}",
            coreType=coreType,
        )
    return constraints

