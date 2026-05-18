from demonstrations.sudoku.constraints._constraint_utils import digit_predicate_core


DEFAULT_ENTROPIC_GROUPS = ((1, 2, 3), (4, 5, 6), (7, 8, 9))


def _category(index, groups):
    digit = index + 1
    for group_index, group in enumerate(groups):
        if digit in group:
            return group_index
    return None


def entropic_window_constraint(posVar1, posVar2, posVar3, sudokuNum=3, groups=DEFAULT_ENTROPIC_GROUPS, coreType=None):
    groups = tuple(tuple(group) for group in groups)
    expected = list(range(len(groups)))

    def is_valid(val1, val2, val3):
        categories = [_category(val, groups) for val in (val1, val2, val3)]
        return None not in categories and sorted(categories) == expected

    return digit_predicate_core(
        [posVar1, posVar2, posVar3],
        is_valid,
        sudokuNum=sudokuNum,
        name=f"entropic_{posVar1}_{posVar2}_{posVar3}",
        coreType=coreType,
    )


def entropic_line_constraints(position_vars, sudokuNum=3, groups=DEFAULT_ENTROPIC_GROUPS, prefix="entropic", coreType=None):
    position_vars = list(position_vars)
    return {
        f"{prefix}_{index}": entropic_window_constraint(
            position_vars[index],
            position_vars[index + 1],
            position_vars[index + 2],
            sudokuNum=sudokuNum,
            groups=groups,
            coreType=coreType,
        )
        for index in range(len(position_vars) - 2)
    }
