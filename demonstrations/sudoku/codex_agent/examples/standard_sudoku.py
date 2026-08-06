from demonstrations.sudoku.codex_agent.constraints.sudoku_constraints import get_sudoku_constraints
from tnreason import representation, application
from tnreason.engine import get_dimDict
import json
from pathlib import Path

from experiments.constraint_networks.sudoku_tests.standard_constraints import (
    get_sudoku_constraint_network,
)


def get_assignment_as_CANetwork(num, startAssignment):
    """
    Prepares the Sudoku game with a start assignment as a Computation-Activation Network
    """
    constraints = get_sudoku_constraints(num)

    comCoreDict = application.create_categorical_cores(constraints)
    canParamDict = {evidenceKey: representation.create_basis_core(name=evidenceKey, shape=[2],
                                                                    colors=[evidenceKey],
                                                                    numberTuple=[startAssignment[evidenceKey]])
                      for evidenceKey in startAssignment}
    featureDict = representation.standard_featureDict_from_computationCoreDict(comCoreDict)
    caNet = representation.ComputationActivationNetwork(
        featureDict=featureDict,
        computationCoreDict=comCoreDict,
        canParamDict=canParamDict
    )
    return caNet



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


def get_assignment_as_constraint_network(num, startAssignment):
    """
    Prepares the standard Sudoku game as a constraint-network core dictionary.
    """
    constraint_cores = get_sudoku_constraint_network(num=num)
    dim_dict = get_dimDict(constraint_cores)
    evidence_cores = {
        evidenceKey: representation.create_basis_core(
            name=evidenceKey,
            shape=[dim_dict[evidenceKey]],
            colors=[evidenceKey],
            numberTuple=[startAssignment[evidenceKey]],
        )
        for evidenceKey in startAssignment
    }
    return {
        **constraint_cores,
        **evidence_cores,
    }
