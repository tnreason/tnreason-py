"""
Assign a new variable, do message passing, check whether consistent (either through vanishing messages appearing or through additional check)
"""
from tnreason import engine, representation

import random
import copy

class Backtracker:
    def __init__(self, propagator, assignment):
        self.propagator = propagator

        self.startAssignment = copy.deepcopy(assignment)
        self.currentAssignment = assignment
        self.startCanParamDict = copy.deepcopy(propagator.caNetwork.canParamDict)  # Deep Copy needed Instead?

        self.failedAssignments = []
        self.plateauAssignments = []

    def search(self, maxIterations=10):
        newKeys = list(self.currentAssignment.keys())
        for iteration in range(maxIterations):
            ## For debugging
            #self.propagator.message_count = 0
            #print("## Current message queue: {}".format(self.propagator.messageQueue))
            #self.propagator.add_affected_directions(newKeys)
            #print("## After message queue: {}".format(self.propagator.messageQueue))
            try:
                print("Iteration {}: Propagation starts with keys {}.".format(iteration + 1, newKeys))
                self.propagator.propagate_until_convergence(nonTrivialFeatureKeys=newKeys, verbose=False)
            except Exception as e:
                print("Propagation ended with exception: {}".format(e))
                self.failedAssignments.append(self.currentAssignment)
                self.reinitialize()
                print("Reinitializing due to inconsistency.")
            else:
                solutionEvidence = meanParams_to_evidence(self.propagator.meanParamDict)
                self.plateauAssignments.append((self.currentAssignment, solutionEvidence))
                self.currentAssignment = solutionEvidence
                print("Propagation without inconsistency. Current assignment length {}.".format(
                    len(self.currentAssignment)))

            try:
                extendAssignment, featureKey = self.guess_extend_an_assignment()
            except:
                return meanParams_to_evidence(self.propagator.meanParamDict), iteration + 1
            else:
                self.currentAssignment = {**self.currentAssignment, **extendAssignment}
                self.propagator.caNetwork.canParamDict.update({
                    featureKey: representation.create_basis_core(name=featureKey,
                                                                 shape=self.propagator.caNetwork.featureDict[
                                                                     featureKey].shape,
                                                                 colors=self.propagator.caNetwork.featureDict[
                                                                     featureKey].featureColors,
                                                                 numberTuple=[extendAssignment[key] for key in
                                                                              extendAssignment])
                })
                newKeys = list(extendAssignment.keys())

        print("Max iterations reached.")
        return self.currentAssignment, maxIterations

    def reinitialize(self):
        ### Here the MC Tree Search could be applied
        self.assignment = copy.deepcopy(self.startAssignment)
        self.propagator.caNetwork.canParamDict = copy.deepcopy(self.startCanParamDict)

    def guess_extend_an_assignment(self):
        ### Here some more advanced heurstics could be applied
        candidates = [featureKey for featureKey in self.propagator.caNetwork.featureDict if
                      "hard" in self.propagator.caNetwork.featureDict[featureKey].featureProperties and engine.contract(
                          {"c": self.propagator.caNetwork.canParamDict[featureKey]}, openColors=[])[:] > 1]
        if len(candidates) == 0:
            raise ValueError("Could not find any candidates for new assignment, probably since all are assigned.")

        featureKey = random.choice(candidates)
        guessedAssignment = self.propagator.caNetwork.featureDict[featureKey].guess_assignment(
            self.propagator.caNetwork.canParamDict[featureKey])
        if {**self.currentAssignment, **guessedAssignment} not in self.failedAssignments:
            print("Guessed assignment {} for feature {}.".format(guessedAssignment, featureKey))
            return guessedAssignment, featureKey
        else:
            self.reinitialize()
            print("Guessed assignment {} for feature {} known to fail, reinitializing.".format(guessedAssignment,
                                                                                               featureKey))
            return self.guess_extend_an_assignment()


def meanParams_to_evidence(meanParamsDict): ## Still on Sudoku! Here counting the atoms known to be true
    return {featureKey: 1 for featureKey in meanParamsDict if
            meanParamsDict[featureKey][{featureKey: 0}] == 0 and featureKey.startswith("a")}


if __name__ == "__main__":
    from demonstrations.sudoku import sudoku_forward_mp as sfmp

    from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence

    import pandas as pd

    df = pd.read_parquet(
        '/Users/alexgoessmann/Documents/tnreason/implementation/demonstrations/sudoku/sudoku_bench/sample_data/nikoli100.parquet')
    puzzleNum = 1
    evidenceDict = initial_board_into_evidence(df["initial_board"][puzzleNum])

    propagator = sfmp.get_forward_mp_inferer(3, evidenceDict)
    backtracker = Backtracker(propagator, evidenceDict)
    result, iteration = backtracker.search(maxIterations=10000)

    print("### RESULT ###")
    print(result)
