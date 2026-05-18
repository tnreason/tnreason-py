from itertools import product
from math import prod

from tnreason import engine


DEFAULT_MAX_DENSE_CELLS = 5_000_000


def digit_count(sudokuNum=3):
    return int(sudokuNum) ** 2


def digit_value(index):
    return int(index) + 1


def _check_dense_size(shape, coreType=None, max_dense_cells=DEFAULT_MAX_DENSE_CELLS):
    if max_dense_cells is None:
        return
    if coreType not in (None, "NumpyCore"):
        return
    cell_count = prod(shape)
    if cell_count > max_dense_cells:
        raise ValueError(
            "Refusing to create a dense constraint core with "
            f"{cell_count} entries. Use a decomposed helper or pass a sparse "
            "coreType/max_dense_cells override."
        )


def predicate_core(
    colors,
    shape,
    predicate,
    name="Constraint",
    coreType=None,
    max_dense_cells=DEFAULT_MAX_DENSE_CELLS,
):
    colors = list(colors)
    shape = [int(size) for size in shape]
    _check_dense_size(shape, coreType=coreType, max_dense_cells=max_dense_cells)

    return engine.create_from_slice_iterator(
        shape=shape,
        colors=colors,
        coreType=coreType,
        name=name,
        sliceIterator=(
            (1, {color: index[pos] for pos, color in enumerate(colors)})
            for index in product(*[range(size) for size in shape])
            if predicate(*index)
        ),
    )


def digit_predicate_core(
    colors,
    predicate,
    sudokuNum=3,
    name="Constraint",
    coreType=None,
    max_dense_cells=DEFAULT_MAX_DENSE_CELLS,
):
    count = digit_count(sudokuNum)
    return predicate_core(
        colors=colors,
        shape=[count] * len(colors),
        predicate=predicate,
        name=name,
        coreType=coreType,
        max_dense_cells=max_dense_cells,
    )


def pair_constraint(
    posVar1,
    posVar2,
    predicate,
    sudokuNum=3,
    name="PairConstraint",
    coreType=None,
):
    return digit_predicate_core(
        [posVar1, posVar2],
        predicate,
        sudokuNum=sudokuNum,
        name=name,
        coreType=coreType,
    )


def line_adjacent_constraints(position_vars, core_factory, prefix):
    constraints = {}
    for index in range(len(position_vars) - 1):
        constraints[f"{prefix}_{index}"] = core_factory(position_vars[index], position_vars[index + 1])
    return constraints


def sum_values(indices):
    return sum(digit_value(index) for index in indices)


def product_values(indices):
    result = 1
    for index in indices:
        result *= digit_value(index)
    return result


def sum_to_hidden_core(
    sum_var,
    position_vars,
    min_sum,
    max_sum,
    sudokuNum=3,
    name="SumToHidden",
    coreType=None,
    max_dense_cells=DEFAULT_MAX_DENSE_CELLS,
):
    position_vars = list(position_vars)
    colors = [sum_var] + position_vars
    shape = [max_sum - min_sum + 1] + [digit_count(sudokuNum)] * len(position_vars)
    return predicate_core(
        colors=colors,
        shape=shape,
        predicate=lambda sum_index, *values: min_sum + sum_index == sum_values(values),
        name=name,
        coreType=coreType,
        max_dense_cells=max_dense_cells,
    )


def product_to_hidden_core(
    product_var,
    position_vars,
    product_values_domain,
    sudokuNum=3,
    name="ProductToHidden",
    coreType=None,
    max_dense_cells=DEFAULT_MAX_DENSE_CELLS,
):
    position_vars = list(position_vars)
    product_values_domain = tuple(int(value) for value in product_values_domain)
    colors = [product_var] + position_vars
    shape = [len(product_values_domain)] + [digit_count(sudokuNum)] * len(position_vars)
    return predicate_core(
        colors=colors,
        shape=shape,
        predicate=lambda product_index, *values: product_values_domain[product_index] == product_values(values),
        name=name,
        coreType=coreType,
        max_dense_cells=max_dense_cells,
    )
