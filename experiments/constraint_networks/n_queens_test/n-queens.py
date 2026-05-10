from tnreason import application
from experiments.constraint_networks import forward_chaining as fc

import numpy as np
from matplotlib import pyplot as plt

def get_queens_constraints(n=3):
    ## Rows
    return {**{"row_" + str(i): ["q_" + str(i) + "_" + str(j) for j in range(n)] for i in range(n)},
            **{"col_" + str(j): ["q_" + str(i) + "_" + str(j) for i in range(n)] for j in range(n)}}


def get_queens_cn(n=3):
    return application.create_categorical_cores(get_queens_constraints(n=n))


def get_queen_positions(solCoreDict, n=3):
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
    plt.imshow(queensPositions, cmap= "binary", vmin=0, vmax=1)
    plt.xticks(range(size), range(size))
    plt.yticks(range(size), range(size))
    plt.title("Random Assignment to the {} queens problem".format(size), fontsize=15)
    plt.show()


if __name__ == "__main__":
    testN = 10
    success, solCoreDict = fc.backtrack(get_queens_cn(n=testN))

    solArray = get_queen_positions(solCoreDict,n=testN)
    draw_positions(solArray)