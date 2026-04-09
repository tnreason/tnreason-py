import numpy as np

def get_variable_num(sudokuNum):
    ## Number of variables in Sudoku instance, for verification
    return sudokuNum ** 6 + 4 * sudokuNum ** 4

def extract_solutionArray_from_coreDict(coresDict, sudokuNum=3):
    solArray = np.empty([sudokuNum ** 2, sudokuNum ** 2])
    for r_0 in range(sudokuNum):
        for r_1 in range(sudokuNum):
            for c_0 in range(sudokuNum):
                for c_1 in range(sudokuNum):
                    if "pos_" + str(r_0) + "_" + str(r_1) + "_" + str(c_0) + "_" + str(
                            c_1) + "_core" in coresDict:
                        solArray[r_0 * sudokuNum + r_1, c_0 * sudokuNum + c_1] = inv_one_hot(
                            coresDict["pos_" + str(r_0) + "_" + str(r_1) + "_" + str(c_0) + "_" + str(
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