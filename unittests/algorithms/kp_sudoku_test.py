import unittest

from tnreason import reasoning
from tnreason import representation
from tnreason import application

import numpy as np


## from examples/sudoku/representation

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


## From expectation_propagation file

def get_featureDict_computationCoresDict_clusterDict(num):
    constraints = get_sudoku_constraints(num)

    atomVariables = ["a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n)
                     for r1 in range(num) for r2 in range(num) for c1 in range(num) for c2 in range(num) for n in
                     range(num ** 2)]

    featureDict = {
        **{atomKey: representation.HardPartitionFeature(featureColors=[atomKey], affectedComputationCores={}) for
           atomKey in
           atomVariables},  # Atomic Variable Features
        **{categoricalKey: representation.HardPartitionFeature(featureColors=[categoricalKey],
                                                               affectedComputationCores=[], shape=[num ** 2])
           for categoricalKey in constraints},
        ## Use empty features instead, to avoid multiple activation cores
        **{categoricalKey + "_" + atomKey: representation.HardPartitionFeature(featureColors=[categoricalKey, atomKey],
                                                                               shape=[num ** 2, 2],
                                                                               affectedComputationCores=[
                                                                                   categoricalKey + "_" + atomKey + representation.suf.atoCoreSuf])
           for categoricalKey in constraints for atomKey in constraints[categoricalKey]}
    }

    computationCores = application.create_categorical_cores(constraints)

    clusterDict = {
        **{atomKey: [atomKey] for atomKey in atomVariables},
        **{categoricalKey: [categoricalKey] for categoricalKey in constraints},
        **{categoricalKey + "_" + atomKey: [categoricalKey, atomKey, categoricalKey + "_" + atomKey]
           for categoricalKey in constraints for atomKey in constraints[categoricalKey]}
    }
    return featureDict, computationCores, clusterDict


## Known assignments into canParams and extraction
def evidence_into_canParams(evidenceDict):
    """
    use, that evidenceKey is a colorKey and a featureKey
    """
    return {evidenceKey: representation.create_basis_core(name=evidenceKey, shape=[2], colors=[evidenceKey],
                                                                                       numberTuple=[evidenceDict[evidenceKey]]) for evidenceKey in
            evidenceDict}


def canParams_into_evidence(meanParamsDict):
    """
    Extraction so far only from atom features, categorical features would require basis vector checks
    """
    return {featureKey: 1 for featureKey in meanParamsDict if
            meanParamsDict[featureKey][{featureKey: 0}] == 0 and featureKey.startswith("a")}


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


class SudokuTest(unittest.TestCase):
    def test_num2(self):
        num = 2

        evidenceDict = {
            "a_0_1_0_0_1": 1,
            "a_0_0_0_1_0": 1,
            "a_0_1_1_0_3": 1,
            "a_1_0_0_0_2": 1,
            "a_0_0_1_0_2": 1
        }

        ## Representation ##

        featureDict, computationCores, clusterDict = get_featureDict_computationCoresDict_clusterDict(num=num)
        caNetwork = representation.ComputationActivationNetwork(
            featureDict=featureDict,
            computationCoreDict=computationCores,
            canParamDict=evidence_into_canParams(evidenceDict)
        )

        ## Reasoning ##

        propagator = reasoning.get_inferer("ExpectationPropagator")(
            caNetwork=caNetwork,
            clusterDict=clusterDict,
            meanParamDict=dict()
        )

        propagator.propagate_until_convergence(evidenceDict.keys())
        solution_array = evidenceDict_to_array(canParams_into_evidence(propagator.meanParamDict), num=num)

        self.assertTrue(solution_array[0, 0] == 4)
        self.assertTrue(solution_array[0, 1] == 1)
        self.assertTrue(solution_array[1, 0] == 2)
        self.assertTrue(solution_array[1, 1] == 3)
        self.assertTrue(solution_array[0, 2] == 3)
        self.assertTrue(solution_array[0, 3] == 2)
        self.assertTrue(solution_array[1, 2] == 4)

    def num3(self):  # Not used in unittests since taking too long! (for usage rename it to test_num3)
        num = 3

        evidenceDict = {
            "a_0_1_0_0_1": 1,
            "a_0_0_0_1_0": 1,
            "a_0_1_1_0_3": 1,
            "a_1_0_0_0_2": 1,
            "a_0_0_1_0_2": 1
        }

        ## Representation ##

        featureDict, computationCores, clusterDict = get_featureDict_computationCoresDict_clusterDict(num=num)
        caNetwork = representation.ComputationActivationNetwork(
            featureDict=featureDict,
            computationCoreDict=computationCores,
            canParamDict=evidence_into_canParams(evidenceDict)
        )

        ## Reasoning ##

        propagator = reasoning.get_inferer("ExpectationPropagator")(
            caNetwork=caNetwork,
            clusterDict=clusterDict,
            meanParamDict=dict()
        )

        propagator.propagate_until_convergence(evidenceDict.keys(), maxMessageCount=100)
        solution_array = evidenceDict_to_array(canParams_into_evidence(propagator.meanParamDict), num=num)

        # Already known from evidenceDict
        self.assertTrue(solution_array[0, 3] == 3)
        self.assertTrue(solution_array[1, 0] == 2)
        self.assertTrue(solution_array[1, 3] == 4)
        self.assertTrue(solution_array[3, 0] == 3)

        ## Direct Solution with ConstraintPropagator:
        # structureCores = representation.create_categorical_cores(get_sudoku_constraints(num=num))
        # preEvidence = {
        #     "a_0_1_0_0_1": 1,
        #     "a_0_0_0_1_0": 1,
        #     "a_0_1_1_0_3": 1,
        #     "a_1_0_0_0_2": 1,
        #     "a_0_0_1_0_2": 1
        # }
        # propagator = reasoning.ConstraintPropagator(
        #     {**structureCores,
        #      **representation.create_evidence_cores(preEvidence)},
        #     verbose=False
        # )
        # propagator.propagate_cores()
        # assignmentDict = propagator.find_assignments()
        #
        # self.assertTrue(assignmentDict["square_1_0_0"] == num)
        # self.assertTrue(not assignmentDict["a_0_0_0_0_1"])


if __name__ == "__main__":
    unittest.main()
