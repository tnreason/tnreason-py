from tnreason import representation, application

import json
import numpy as np
from tnreason.engine.workload_to_numpy import NumpyCore
"""
We here provide the main Sudoku constraints by dictionaries of categorical constraints.
These constraints can be represented by tensor networks using tnreason.representation.create_categorical_cores(constraints).
"""
from pathlib import Path

def encode_core(obj):
    if isinstance(obj, NumpyCore):
        return {
            "coreType": obj.coreType,
            "name": obj.name,
            "colors": list(obj.colors),
            "shape": [int(s) for s in obj.shape],
            "values": obj.values.tolist(),
        }
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):  # numpy scalars
        return obj.item()
    raise TypeError(f"{type(obj).__name__} not JSON serializable")

def get_assignment_as_CANetwork_load(num, startAssignment):
    """
    Prepares the Sudoku game with a start assignment as a Computation-Activation Network
    """
    # constraints = get_sudoku_constraints(num)
    #
    # atomVariables = ["a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n)
    #                  for r1 in range(num) for r2 in range(num) for c1 in range(num) for c2 in range(num) for n in
    #                  range(num ** 2)]
    with Path(f"demonstrations/sudoku/examples/n={num}.txt").open("r") as f:
        constraints = json.load(f)
    comCoreDict = application.create_categorical_cores(constraints)
    
    canParamDict = {evidenceKey: representation.create_basis_core(name=evidenceKey, shape=[2],
                            colors=[evidenceKey], numberTuple=[startAssignment[evidenceKey]])
                    for evidenceKey in startAssignment}
    featureDict = representation.standard_featureDict_from_computationCoreDict(comCoreDict)

    caNet = representation.ComputationActivationNetwork(
        featureDict=featureDict,
        computationCoreDict=comCoreDict,
        canParamDict=canParamDict
    )
    return caNet

def get_assignment_as_CANetwork(num, startAssignment):
    """
    Prepares the Sudoku game with a start assignment as a Computation-Activation Network
    """
    constraints = get_sudoku_constraints(num)
    
    # atomVariables = ["a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n)
    #                  for r1 in range(num) for r2 in range(num) for c1 in range(num) for c2 in range(num) for n in
    #                  range(num ** 2)]
    
    # path = Path(f"demonstrations/sudoku/examples/n={num}.txt")
    # path.write_text(json.dumps(constraints, indent=2, sort_keys=True, default=encode_core))

    comCoreDict = application.create_categorical_cores(constraints)
    # path = Path(f"demonstrations/sudoku/examples/n={num}_comCoreDict.txt")
    # path.write_text(json.dumps(comCoreDict, indent=2, sort_keys=True, default=encode_core))
    
    canParamDict = {evidenceKey: representation.create_basis_core(name=evidenceKey, shape=[2],
                                                                    colors=[evidenceKey],
                                                                    numberTuple=[startAssignment[evidenceKey]])
                      for evidenceKey in startAssignment}
    # path = Path(f"demonstrations/sudoku/examples/n={num}_canParamDict.txt")
    # path.write_text(json.dumps(canParamDict, indent=2, sort_keys=True, default=encode_core))

    featureDict = representation.standard_featureDict_from_computationCoreDict(comCoreDict)
    # path = Path(f"demonstrations/sudoku/examples/n={num}_featureDict.txt")
    # path.write_text(json.dumps({key: str(type(featureDict[key])) for key in featureDict.keys()}, indent=2, sort_keys=True, default=encode_core))

    caNet = representation.ComputationActivationNetwork(
        featureDict=featureDict,
        computationCoreDict=comCoreDict,
        canParamDict=canParamDict
    )
    return caNet

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
