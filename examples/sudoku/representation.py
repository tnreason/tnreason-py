from tnreason import reasoning
from tnreason import representation

import numpy as np


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


#    return categoricalConstraints


def get_position_constraints(num=3):
    """
    Exactly one number is assigned at each position.
    """
    return {"pos_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2): [
        "a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n)
        for n in range(num ** 2)
    ] for r1 in range(num) for r2 in range(num) for c1 in range(num) for c2 in range(num)}


def evidenceDict_to_array(evidenceDict, num):
    array = np.zeros(shape=(num ** 2, num ** 2))
    for variableKey in evidenceDict:
        varDecom = variableKey.split("_")
        if varDecom[0] == "a" and evidenceDict[variableKey] == 1:
            # print(variableKey)
            array[int(varDecom[1]) * num + int(varDecom[2]), int(varDecom[3]) * num + int(varDecom[4])] = int(
                varDecom[5]) + 1
        elif varDecom[0] == "pos":
            array[int(varDecom[1]) * num + int(varDecom[2]), int(varDecom[3]) * num + int(varDecom[4])] = evidenceDict[
                                                                                                              variableKey] + 1
    return array


## Visualization
def evidence_to_array(evidenceDict, num, verbose=False):
    array = np.empty((num ** 2, num ** 2))
    for r1 in range(num):
        for r2 in range(num):
            for c1 in range(num):
                for c2 in range(num):
                    catVarKey = "pos_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2)
                    if catVarKey in evidenceDict:
                        if verbose:
                            print(
                                "Position {} known to be {}".format(
                                    str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2),
                                    evidenceDict[catVarKey] + 1))
                        array[r1 * num + r2, c1 * num + c2] = evidenceDict[catVarKey] + 1
                    else:
                        array[r1 * num + r2, c1 * num + c2] = 0
    return array