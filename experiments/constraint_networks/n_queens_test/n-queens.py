from tnreason import application, engine
from experiments.constraint_networks import unit_propagation as fc

from tnreason.application.categoricals_to_cores import create_at_most_one_constraints

import numpy as np
from matplotlib import pyplot as plt


def get_rook_constraints(n=3):
    return {**{"row_" + str(i): ["q_" + str(i) + "_" + str(j) for j in range(n)] for i in range(n)},
            **{"col_" + str(j): ["q_" + str(i) + "_" + str(j) for i in range(n)] for j in range(n)}}


def get_diagonal_constraints(n=3):
    return {**{"rdia_" + str(l): ["q_" + str(l + i) + "_" + str(i) for i in range(max(0, -l), min(n, n - l))] for l in
               range(-(n - 2), (n - 1))},
            **{"ldia_" + str(l): ["q_" + str(l - i) + "_" + str(i) for i in range(max(0, l - (n - 1)), min(n, l + 1))]
               for l
               in
               range(1, 2 * n - 2)}}


def get_queens_constraints(n=3):
    return {**get_rook_constraints(n=n),
            **get_diagonal_constraints(n=n)}


def get_queens_cn(n=3):
    return {**application.create_categorical_cores(get_rook_constraints(n=n)),
            **create_at_most_one_constraints(get_diagonal_constraints(n=n))}


def get_positions(solCoreDict, n=3):
    queensPositions = np.zeros(shape=(n, n))
    for colPos in range(n):
        for rowPos in range(n):
            if solCoreDict["q_" + str(rowPos) + "_" + str(colPos) + "_core"][
                {"q_" + str(rowPos) + "_" + str(colPos): 0}] == 0:
                queensPositions[rowPos, colPos] = 1
            else:
                queensPositions[rowPos, colPos] = 0
    return queensPositions


def draw_positions(queensPositions):
    size = queensPositions.shape[0]
    plt.imshow(queensPositions, cmap="binary", vmin=0, vmax=1)
    plt.xticks(range(size), range(size))
    plt.yticks(range(size), range(size))
    plt.title("Random assignment to the {} rooks problem".format(size), fontsize=15)
    plt.show()


if __name__ == "__main__":

    def calculate_possibilities(n):
        from tnreason import engine
        import numpy as np
        return np.linalg.norm(engine.contract(coreDict=get_queens_cn(n),
                                              openColors=["q_0_0"],
                                              contractionMethod="PgmpyVariableEliminator").values)


    assert 1 == calculate_possibilities(n=1)
    assert 0 == calculate_possibilities(n=2)
    assert 0 == calculate_possibilities(n=3)
    assert 2 == calculate_possibilities(n=4)

    ## Solving rooks by backtracking
    testN = 20
    success, solCoreDict = fc.backtrack(application.create_categorical_cores(get_rook_constraints(n=testN)))

    solArray = get_positions(solCoreDict, n=testN)
    draw_positions(solArray)
