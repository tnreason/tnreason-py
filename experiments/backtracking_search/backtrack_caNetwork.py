"""
Assign a new variable, do message passing, check whether consistent (either through vanishing messages appearing or through additional check)
"""
from tnreason import engine, representation, application
from demonstrations.sudoku import visualization as vis

import random
import copy

from demonstrations.sudoku import sudoku_forward_mp as sfmp

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

            start_array = vis.evidence_to_array(self.currentAssignment, num=self.num)
            vis.visualize_sudoku(start_array.astype(int), number=self.num, label=f"Assignment {iteration + 1}")

            self.extend_clusters()

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
        # Detect locked candidates_box_line constraints
        locked_info = self.detect_locked_candidates_box_line()
        locked_inferenceClusters = {"_".join(vars[::-1]): vars + ["_".join(vars[::-1])] for vars in locked_info}

        while len(locked_inferenceClusters) > 0:
            # Check, if we already know those constraints, if yes, remove them from the new constraints to add
            overlap = locked_inferenceClusters.keys() & self.propagator.inferenceClusters.keys()
            for key in overlap:
                del locked_inferenceClusters[key]
            print(f"Locked candidates to add as new constraints: {locked_inferenceClusters.keys()}")

            if len(locked_inferenceClusters) > 0:
                # Repropagate if we have found new constraints
                new_inferenceClusters = {**self.propagator.inferenceClusters, **locked_inferenceClusters}
                self.propagator.update_inferenceClusters(new_inferenceClusters)
                self.try_to_propagate(newKeys=[color for colors in locked_info for color in colors])  
                # self.propagator.propagate_until_convergence(nonTrivialFeatureKeys=[color for colors in locked_info for color in colors], verbose=False)
                
                # Search for it again, since we might have added new constraints that allow to detect more locked candidates
                locked_info = self.detect_locked_candidates_box_line()
                locked_inferenceClusters = {"_".join(vars[::-1]): vars + ["_".join(vars[::-1])] for vars in locked_info}
        
        # Update evidence after repropagation
        solutionEvidence = meanParams_to_evidence(self.propagator.meanParamDict)
        self.currentAssignment = solutionEvidence

    def reinitialize(self):
        ### Here the MC Tree Search could be applied
        self.assignment = copy.deepcopy(self.startAssignment)
        self.propagator.caNetwork.canParamDict = copy.deepcopy(self.startCanParamDict)

    def detect_locked_candidates_box_line(self):
        """
        Detect hidden singles: if in a square, a number can only be in one row or one column.
        Returns list of tuples: (square_key, number, 'row'/'col', row/col_index)
        """
        locked_candidates = []
        num = self.num

        for r1 in range(num):
            for c1 in range(num):
                for n in range(num**2):
                    square_key = f"square_{n}_{r1}_{c1}"
                    # square_constraint = ["a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n) for r2 in range(num) for c2 in range(num)]
                    square_constraint_inds = [(r2,c2) for r2 in range(num) for c2 in range(num)]
                    possible_pos_flat = [i for i in range(num**2) if self.propagator.meanParamDict[square_key][{square_key: i}] > 0.5]
                    possible_positions = [square_constraint_inds[ind] for ind in possible_pos_flat]
                    
                    if not possible_positions:
                        continue
                    
                    # Check rows
                    rows = set(pos[0] for pos in possible_positions)
                    cols = set(pos[1] for pos in possible_positions)
                    if len(rows) == 1 and len(cols) > 1:
                        locked_row = list(rows)[0]
                        locked_candidates.append([square_key,
                                                "row_" + str(n) + "_" + str(r1) + "_" + str(locked_row),
                                                ]+[
                                                "a_" + str(r1) + "_" + str(locked_row) + "_" + str(c1) + "_" + str(c2) + "_" + str(n)
                                                    for c2 in list(cols)
                                                ])
                        print(f"Locked candidates detected: {square_key} number {n} locked in row {r1}_{locked_row}")
                    
                    # Check columns
                    if len(cols) == 1 and len(rows) > 1:
                        locked_col = list(cols)[0]
                        locked_candidates.append([square_key,
                                                "col_" + str(n) + "_" + str(c1) + "_" + str(locked_col),
                                                ]+[
                                                "a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(locked_col) + "_" + str(n)
                                                    for r2 in list(rows)
                                                ])
                        print(f"Locked candidates detected: {square_key} number {n} locked in col {c1}_{locked_col}")
        
        return locked_candidates

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
    
    def guess_extend_an_assignment_(self):
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

        chosen_atom = mcts.pick_action(self.currentAssignment)

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
    from demonstrations.sudoku import sudoku_forward_mp as sfmp

    from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence

    import pandas as pd

    df = pd.read_parquet(
        '/home/schuette/Desktop/AlphaSudoku/tnreason-py-version2(2)/tnreason-py-version2/demonstrations/sudoku/sudoku_bench/sample_data/nikoli100.parquet')
        # '/Users/alexgoessmann/Documents/tnreason/implementation/demonstrations/sudoku/sudoku_bench/sample_data/nikoli100.parquet')
    puzzleNum = 1
    evidenceDict = initial_board_into_evidence(df["initial_board"][puzzleNum])

    backtracker = Backtracker(evidenceDict)
    result, iteration = backtracker.search(maxIterations=10000)

    print("### RESULT ###")
    print(result)
