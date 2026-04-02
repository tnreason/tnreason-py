from experiments.constraint_networks import forward_chaining as fc

from experiments.constraint_networks.sudoku_tests import standard_constraints as sc
from experiments.constraint_networks.sudoku_tests import read_evidence as re

import numpy as np


def fc_test_on_puzzle(puzzlePos):
    constraintDict = {
        **sc.get_sudoku_constraint_network(num=3),
        **re.read_evidence(puzzlePos)
    }

    chainer = fc.GenericForwardChaining(
        constraintDict
    )
    chainer.propagate_all_singleNodeEdges()

    return len(chainer.disentangledNodes) == get_variable_num(3), chainer.cn


def get_variable_num(sudokuNum):
    ## Number of variables in Sudoku instance
    return sudokuNum ** 6 + 4 * sudokuNum ** 4


def extract_solutionArray_from_constraintNetwork(constraintNetwork, sudokuNum=3):
    solArray = np.empty([sudokuNum ** 2, sudokuNum ** 2])
    for r_0 in range(sudokuNum):
        for r_1 in range(sudokuNum):
            for c_0 in range(sudokuNum):
                for c_1 in range(sudokuNum):
                    #                    print(constraintNetwork)
                    if "pos_" + str(r_0) + "_" + str(r_1) + "_" + str(c_0) + "_" + str(
                            c_1) + "_core" in constraintNetwork.coresDict:
                        #                        print(constraintNetwork.coresDict["pos_" + str(r_0) + "_" + str(r_1) + "_" + str(c_0) + "_" + str(
                        #                            c_1) + "_core"].values)
                        solArray[r_0 * sudokuNum + r_1, c_0 * sudokuNum + c_1] = inv_one_hot(
                            constraintNetwork.coresDict["pos_" + str(r_0) + "_" + str(r_1) + "_" + str(c_0) + "_" + str(
                                c_1) + "_core"]) + 1
    return solArray


def inv_one_hot(vect):
    foundPos = False
    for idx in np.ndindex(vect.shape):
        if vect[idx] != 0:
            if foundPos:
                # Then not a basis vector
                return -1
            foundPos = True
            pos = idx
    return pos[0]


def full_run():
    sucCount = 0
    for pos in range(100):
        print(f"## Puzzle {pos} ##")
        success, resCn = fc_test_on_puzzle(pos)
        print(f"Success: {success}")
        if success:
            sucCount += 1
            print("Solution verified: {}".format(
                np.array_equal(extract_solutionArray_from_constraintNetwork(resCn), re.get_solution_array(pos))))
    #                extract_solutionArray_from_constraintNetwork(resCn)==re.get_solution_array(pos)))
    print("#######")
    print(f"Summary: {sucCount} of 100 solved.")
    return sucCount


if __name__ == "__main__":
    # success, resCn = fc_test_on_puzzle(0)
    # print(extract_solutionArray_from_constraintNetwork(resCn))
    # print(re.get_solution_array(0))

    full_run()
