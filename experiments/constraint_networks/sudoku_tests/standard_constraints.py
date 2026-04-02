def get_sudoku_constraints(num=3):
    return {**get_column_constraints(num),
            **get_row_constraints(num),
            **get_squares_constraints(num),
            **get_position_constraints(num)}


def get_column_constraints(num=3):
    """
    In each column each number appears exactly once.
    """
    return {"col_" + str(n) + "_" + str(c1) + "_" + str(c2): [
        "a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n) for r1 in range(num) for r2 in
        range(num)] for c1 in range(num) for c2 in range(num) for n in range(num ** 2)}


def get_row_constraints(num=3):
    """
    In each row each number appears exactly once.
    """
    return {"row_" + str(n) + "_" + str(r1) + "_" + str(r2): [
        "a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n) for c1 in range(num) for c2 in
        range(num)
    ] for r1 in range(num) for r2 in range(num) for n in
        range(num ** 2)}


def get_squares_constraints(num=3):
    """
    In each square each number appears exactly once.
    """
    return {"square_" + str(n) + "_" + str(r1) + "_" + str(c1): [
        "a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n)
        for r2 in range(num) for c2 in range(num)
    ]
        for r1 in range(num) for c1 in range(num) for n in range(num ** 2)}


def get_position_constraints(num=3):
    """
    Exactly one number is assigned at each position.
    """
    return {"pos_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2): [
        "a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n)
        for n in range(num ** 2)
    ] for r1 in range(num) for r2 in range(num) for c1 in range(num) for c2 in range(num)}


from tnreason import engine


def get_sudoku_constraint_network(num=3):
    catDict = get_sudoku_constraints(num=num)
    return {catKey + "_" + atomKey: engine.create_from_slice_iterator(colors=[atomKey, catKey],
                                                                      shape=[2, len(catDict[catKey])],
                                                                      sliceIterator=[(1, {atomKey: 0}),
                                                                                     (-1, {atomKey: 0, catKey: pos}),
                                                                                     (1, {atomKey: 1, catKey: pos})])
            for catKey in catDict for pos, atomKey in enumerate(catDict[catKey])}


def get_constraint_matrix(catKey, atomKey, catDim, pos):
    return engine.create_from_slice_iterator(colors=[atomKey, catKey], shape=[2, catDim],
                                             sliceIterator=[(1, {atomKey: 0}), (-1, {atomKey: 0, catKey: pos}),
                                                            (1, {atomKey: 1, catKey: pos})])

if __name__ == "__main__":
    print(get_sudoku_constraint_network(num=2))
