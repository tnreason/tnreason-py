"""
Assign a new variable, do message passing, check whether consistent (either through vanishing messages appearing or through additional check)
"""
from hashlib import new
from tnreason import engine, representation, application
from demonstrations.sudoku import visualization as vis

import random
import copy

from demonstrations.sudoku import sudoku_forward_mp as sfmp
import demonstrations.sudoku.inferenceClusters as infClusters

class Backtracker:
    def __init__(self, assignment):
        self.num=3
        propagator = sfmp.get_forward_mp_inferer(self.num, assignment)
        self.propagator = propagator

        self.startAssignment = copy.deepcopy(assignment)
        self.currentAssignment = assignment
        self.startCanParamDict = copy.deepcopy(propagator.caNetwork.canParamDict)  # Deep Copy needed Instead?

        self.failedAssignments = []
        self.plateauAssignments = []

    def search(self, maxIterations=10):
        newKeys = list(self.currentAssignment.keys())

        start_array = vis.evidence_to_array(self.currentAssignment, num=self.num)
        vis.visualize_sudoku(start_array.astype(int), number=self.num, label=f"Assignment {0}")

        for iteration in range(maxIterations):
            ## For debugging
            #self.propagator.message_count = 0
            #print("## Current message queue: {}".format(self.propagator.messageQueue))
            #self.propagator.add_affected_directions(newKeys)
            #print("## After message queue: {}".format(self.propagator.messageQueue))

            print("Iteration {}: Propagation starts with keys {}.".format(iteration + 1, newKeys))
            self.try_to_propagate(newKeys)
            self.extend_clusters()


            start_array = vis.evidence_to_array(self.currentAssignment, num=self.num)
            vis.visualize_sudoku(start_array.astype(int), number=self.num, label=f"Assignment {iteration + 1}")

            
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

    def try_to_propagate(self, newKeys):
        try:
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



    def extend_clusters(self):
        new_info = self.detect_new_rules()
        new_inferenceClusters = {"_".join(vars[::-1]): vars + ["_".join(vars[::-1])] for vars in new_info}

        while len(new_inferenceClusters) > 0:
            # Check, if we already know those constraints, if yes, remove them from the new constraints to add
            overlap = new_inferenceClusters.keys() & self.propagator.inferenceClusters.keys()
            for key in overlap:
                del new_inferenceClusters[key]
            print(f"New candidates to add as new constraints: {new_inferenceClusters.keys()}")

            if len(new_inferenceClusters) > 0:
                # Repropagate if we have found new constraints
                new_inferenceClusters = {**self.propagator.inferenceClusters, **new_inferenceClusters}
                self.propagator.update_inferenceClusters(new_inferenceClusters)
                self.try_to_propagate(newKeys=[color for colors in new_info for color in colors])
                
                # Search for it again, since we might have added new constraints that allow to detect more locked candidates
                new_info = self.detect_new_rules() # Here more types of constraints could be added, e.g. naked pairs, pointing pairs, etc.
                new_inferenceClusters = {"_".join(vars[::-1]): vars + ["_".join(vars[::-1])] for vars in new_info}
        
        # Update evidence after repropagation
        solutionEvidence = meanParams_to_evidence(self.propagator.meanParamDict)
        self.currentAssignment = solutionEvidence

    def reinitialize(self):
        ### Here the MC Tree Search could be applied
        self.assignment = copy.deepcopy(self.startAssignment)
        self.propagator.caNetwork.canParamDict = copy.deepcopy(self.startCanParamDict)

    def detect_new_rules(self):
        """
        Detect new constraints based on the current mean parameters. For example, locked candidates in Sudoku.
        Returns list of variable combinations that form new constraints, e.g. for locked candidates: (square_key, number, 'row'/'col', row/col_index)
        """
        # Detect locked candidates_box_line constraints
        locked_square = infClusters.detect_locked_candidates_square(self.propagator, num=self.num)
        locked_row = infClusters.detect_locked_candidates_row(self.propagator, num=self.num)
        locked_col = infClusters.detect_locked_candidates_col(self.propagator, num=self.num)
        hidden_pairs = infClusters.detect_hidden_pairs(self.propagator, num=self.num)
        naked_pairs = infClusters.detect_naked_pairs(self.propagator, num=self.num)
        # Here more types of constraints could be added, e.g. naked pairs, pointing pairs, etc.
        return locked_square+locked_row+locked_col+hidden_pairs+naked_pairs

    def guess_extend_an_assignment_(self):
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
    
    def guess_extend_an_assignment(self):
        from demonstrations.sudoku import sudoku_forward_mp as sfmp
        from experiments.backtracking_search.alphasudoku_mcts import MpBackend, AlphaSudokuMCTS

        num = self.num  # <-- set to your Sudoku block size (3 for 9x9)

        mp = MpBackend(num=num, get_inferer=sfmp.get_forward_mp_inferer)
        mcts = AlphaSudokuMCTS(
            num=num,
            mp_backend=mp,
            simulations=400,      # increase for harder puzzles
            top_k_actions=6,      # branching control
            c_puct=1.5,
        )

        chosen_atom = mcts.pick_action(
            self.currentAssignment,
            inference_clusters=self.propagator.inferenceClusters
        )

        # Backtracker expects (extendAssignment, featureKey)
        # Here featureKey is the variable we assign (the atom itself)
        extendAssignment = {chosen_atom: 1}
        featureKey = chosen_atom
        print(f"MCTS guessed {extendAssignment} for feature {featureKey}.")
        return extendAssignment, featureKey



def meanParams_to_evidence(meanParamsDict): ## Still on Sudoku! Here counting the atoms known to be true
    return {featureKey: 1 for featureKey in meanParamsDict if
            meanParamsDict[featureKey][{featureKey: 0}] == 0 and featureKey.startswith("a")}


if __name__ == "__main__":
    from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence
    from check_caNetwork import check_consistency
    import pandas as pd
    import json

    # Load puzzle
    puzzleNum = 1
    path = '/home/schuette/Desktop/AlphaSudoku/tnreason-py-version2(2)/tnreason-py-version2/demonstrations/sudoku/sudoku_bench/sample_data/'
    # path = '/Users/alexgoessmann/Documents/tnreason/implementation/demonstrations/sudoku/sudoku_bench/sample_data/')
    df = pd.read_parquet(f'{path}nikoli100.parquet')
    evidenceDict = initial_board_into_evidence(df["initial_board"][puzzleNum])

    # Solve puzzle with backtracking search
    backtracker = Backtracker(evidenceDict)
    result, iteration = backtracker.search(maxIterations=10000)
    print("### RESULT ###")
    print(result)

    # Save result
    with open(f"{path}backtracking_result_{puzzleNum}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Check result by contraction
    from demonstrations.sudoku.constraints import sudoku_constraints as rep
    caNetwork = rep.get_assignment_as_CANetwork(num=3, startAssignment=evidenceDict)
    assert check_consistency(caNetwork, result) 
    
