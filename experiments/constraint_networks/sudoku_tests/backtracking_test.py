from experiments.constraint_networks import forward_chaining as fc

from experiments.constraint_networks.sudoku_tests import standard_constraints as sc
from experiments.constraint_networks.sudoku_tests import read_evidence as re
from experiments.constraint_networks.sudoku_tests import extract_solution as es

import numpy as np


def backtrack_test_on_puzzle(puzzlePos, verbose=False, maxDepth=3, maxRestarts=100):
    constraintDict = {
        **sc.get_sudoku_constraint_network(num=3),
        **re.read_evidence(puzzlePos)
    }
    return fc.backtrack_with_restarts(constraintDict, maxDepth=maxDepth, maxRestarts=maxRestarts, verbose=verbose)


def backtrack_full_run(untilPuzzle=100):
    sucCount = 0
    for pos in range(untilPuzzle):
        print(f"## Puzzle {pos} ##")
        success, resCores, restartsDone = backtrack_test_on_puzzle(pos)
        print(f"Success after {restartsDone} restarts: {success}")
        if success:
            sucCount += 1
            assert np.array_equal(es.extract_solutionArray_from_coreDict(resCores), re.get_solution_array(pos))
    print("#######")
    print(f"Summary: {sucCount} of {untilPuzzle} solved.")
    return sucCount


if __name__ == "__main__":
    backtrack_full_run()