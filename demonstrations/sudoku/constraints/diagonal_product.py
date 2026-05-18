from demonstrations.sudoku.constraints._constraint_utils import digit_predicate_core, product_to_hidden_core, product_values


def diagonal_product_constraint(position_vars, target_product, sudokuNum=3, coreType=None, max_dense_cells=5_000_000):
    position_vars = list(position_vars)
    return digit_predicate_core(
        position_vars,
        lambda *values: product_values(values) == target_product,
        sudokuNum=sudokuNum,
        name="diagonal_product_" + "_".join(position_vars),
        coreType=coreType,
        max_dense_cells=max_dense_cells,
    )


def diagonal_product_to_hidden_constraint(
    position_vars,
    product_var,
    product_values_domain,
    sudokuNum=3,
    coreType=None,
    max_dense_cells=5_000_000,
):
    return product_to_hidden_core(
        product_var=product_var,
        position_vars=position_vars,
        product_values_domain=product_values_domain,
        sudokuNum=sudokuNum,
        name="diagonal_product_hidden_" + "_".join(position_vars),
        coreType=coreType,
        max_dense_cells=max_dense_cells,
    )


def same_product_constraints(diagonals, product_var="diagonal_product", product_values_domain=None, sudokuNum=3, coreType=None):
    if product_values_domain is None:
        raise ValueError("product_values_domain is required for hidden-product constraints")
    return {
        f"{product_var}_{index}": diagonal_product_to_hidden_constraint(
            diagonal,
            product_var,
            product_values_domain,
            sudokuNum=sudokuNum,
            coreType=coreType,
        )
        for index, diagonal in enumerate(diagonals)
    }

