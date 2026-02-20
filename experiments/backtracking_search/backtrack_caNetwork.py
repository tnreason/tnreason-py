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

    def extend_clusters(self):
        # Detect and add hidden single constraints
        hidden_info = self.detect_square_to_row_col()

        new_keys_from_hidden = []
        for square_key, num, row_or_col, index in hidden_info:
            try:
                constraint_key = self.add_hidden_single_constraint(square_key, num, row_or_col, index)
                new_keys_from_hidden.append(constraint_key)
            except Exception as e:
                print(f"Could not add hidden single constraint: {e}")
        
        # Repropagate if we added new constraints
        if new_keys_from_hidden:
            print(f"Repropagating with {len(new_keys_from_hidden)} new hidden single constraints")
            self.propagator.propagate_until_convergence(nonTrivialFeatureKeys=new_keys_from_hidden, verbose=False)
            # Update evidence after repropagation
            solutionEvidence = meanParams_to_evidence(self.propagator.meanParamDict)
            self.currentAssignment = solutionEvidence





    def reinitialize(self):
        ### Here the MC Tree Search could be applied
        self.assignment = copy.deepcopy(self.startAssignment)
        self.propagator.caNetwork.canParamDict = copy.deepcopy(self.startCanParamDict)

    def detect_square_to_row_col(self):
        """
        Detect hidden singles: if in a square, a number can only be in one row or one column.
        Returns list of tuples: (square_key, number, 'row'/'col', row/col_index)
        """
        hidden_singles = []
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
                        hidden_singles.append((square_key, n, 'row', locked_row))
                        print(f"Hidden single detected: {square_key} number {n} locked in row {r1}_{locked_row}")
                    
                    # Check columns
                    if len(cols) == 1 and len(rows) > 1:
                        locked_col = list(cols)[0]
                        hidden_singles.append((square_key, n, 'col', locked_col))
                        print(f"Hidden single detected: {square_key} number {n} locked in col {c1}_{locked_col}")
        
        return hidden_singles
    
    def add_hidden_single_constraint(self, square_key, number, row_or_col, index):
        """
        Add an inference cluster that links a square constraint to a row/column constraint
        when a number is locked in that row/column within the square.
        """
        # Parse square key: square_{n}_{r1}_{c1}
        parts = square_key.split('_')
        n = int(parts[1])
        r1 = int(parts[2])
        c1 = int(parts[3])
        
        if row_or_col == 'row':
            # Create constraint: locked_row_n_r1_c1_r2
            constraint_key = f"locked_row_{n}_{r1}_{c1}_{index}"
            row_constraint = f"row_{n}_{r1}_{index}"
            
            # Atoms in this row of the square
            atom_vars = []
            for c2 in range(self.num):
                atom_vars.append(f"a_{r1}_{index}_{c1}_{c2}_{n}")
            
            variables = [square_key, row_constraint] + atom_vars
        else:  # col
            # Create constraint: locked_col_n_r1_c1_c2
            constraint_key = f"locked_col_{n}_{r1}_{c1}_{index}"
            col_constraint = f"col_{n}_{c1}_{index}"
            
            # Atoms in this column of the square
            atom_vars = []
            for r2 in range(self.num):
                atom_vars.append(f"a_{r1}_{r2}_{c1}_{index}_{n}")
            
            variables = [square_key, col_constraint] + atom_vars
        
        # Create categorical cores using the application API
        # This creates an "exactly one" constraint among all variables
        categoricalsDict = {constraint_key: variables}
        new_cores = application.create_categorical_cores(categoricalsDict)
        
        # Add the new constraint to the computation core dict
        self.propagator.caNetwork.computationCoreDict.update(new_cores)
        
        # Rebuild inference clusters to include the new constraint
        from tnreason.reasoning.message_passing import standard_clusters_from_computationCoreDict
        messageClusters, inferenceClusters = standard_clusters_from_computationCoreDict(
            self.propagator.caNetwork.computationCoreDict
        )
        # self.propagator.inferenceClusters = inferenceClusters
        # self.propagator.messageClusters = messageClusters

        self.propagator.update_inferenceClusters(inferenceClusters)        
        
        print(f"Added constraint: {constraint_key} with variables: {variables}")
        return constraint_key

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
