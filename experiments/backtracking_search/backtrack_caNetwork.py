"""
Assign a new variable, do message passing, check whether consistent (either through vanishing messages appearing or through additional check)
"""
from tnreason import engine, representation, reasoning

from demonstrations.sudoku import sudoku_forward_mp as sol

import random


class Backtracker:
    def __init__(self, propagator, assignment):
        self.propagator = propagator

        self.startAssignment = assignment
        self.currentAssignment = assignment
        self.startCanParamDict = propagator.caNetwork.canParamDict # Deep Copy?

        self.failedAssignments = []
        self.plateauAssignments = []

    def search(self, maxIterations=10):
        newKeys = list(self.currentAssignment.keys())
        for iteration in range(maxIterations):
            try:
                self.propagator.propagate_until_convergence(nonTrivialFeatureKeys=newKeys)
            except:
                self.failedAssignments.append(self.currentAssignment)
                self.reinitialize()
                print("Iteration {}: Reinitializing due to inconsistency.".format(iteration))
            else:
                solutionEvidence = sol.meanParams_to_evidence(self.propagator.meanParamDict)
                self.plateauAssignments.append((self.currentAssignment, solutionEvidence))
                self.currentAssignment = solutionEvidence
                print("Iteration {}: Current assignment length {}.".format(iteration, len(self.currentAssignment)))

            try:
                newKeys = self.guess_extend_an_assignment()
            except:
                return sol.meanParams_to_evidence(self.propagator.meanParamDict)
            else:
                self.currentAssignment = {**self.currentAssignment, **newKeys}

    def reinitialize(self):
        self.assignment = self.startAssignment
        self.propagator.caNetwork.canParamDict = self.startCanParamDict

    def guess_extend_an_assignment(self):
        candidates = [featureKey for featureKey in self.propagator.caNetwork.featureDict if
                      "hard" in self.propagator.caNetwork.featureDict[featureKey].featureProperties and engine.contract(
                          {"c": self.propagator.caNetwork.canParamDict[featureKey]}, openColors=[])[:] > 1]
        if len(candidates) == 0:
            #print("Solution: {}".format(sol.meanParams_to_evidence(self.propagator.meanParamDict)))
            raise ValueError("Could not find any candidates for new assignment, probably since all are assigned.")

        featureKey = random.choice(candidates)
        guessedAssignment = self.propagator.caNetwork.featureDict[featureKey].guess_assignment(
            self.propagator.caNetwork.canParamDict[featureKey])
        if {**self.currentAssignment, **guessedAssignment} not in self.failedAssignments:
            print("Guessed assignment {} for feature {}.".format(guessedAssignment, featureKey))
            return guessedAssignment
        else:
            self.reinitialize()
            print("Guessed assignment {} for feature {} known to fail, reinitializing.".format(guessedAssignment, featureKey))
            return self.guess_extend_an_assignment()


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
    backtracker.search(maxIterations=10)