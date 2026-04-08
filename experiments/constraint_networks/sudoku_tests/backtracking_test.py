from experiments.constraint_networks import forward_chaining as fc

from experiments.constraint_networks.sudoku_tests import standard_constraints as sc
from experiments.constraint_networks.sudoku_tests import read_evidence as re



def backtrack_test_on_puzzle(puzzlePos):
    constraintDict = {
        **sc.get_sudoku_constraint_network(num=3),
        **re.read_evidence(puzzlePos)
    }
    return fc.backtrack(constraintDict)


if __name__ == "__main__":
    success, resCoreDict = backtrack_test_on_puzzle(1)

