from experiments.constraint_networks import forward_chaining as fc

from experiments.constraint_networks.sudoku_tests import standard_constraints as sc
from experiments.constraint_networks.sudoku_tests import read_evidence as re
from experiments.constraint_networks.sudoku_tests import extract_solution as es

import numpy as np


def fc_test_on_puzzle(puzzlePos):
    constraintDict = {
        **sc.get_sudoku_constraint_network(num=3),
        **re.read_evidence(puzzlePos)
    }
    chainer = fc.GenericForwardChaining(
        constraintDict
    )
    chainer.propagate_all_singleNodeEdges()

    return len(chainer.disentangledNodes) == es.get_variable_num(3), chainer.cn

def full_run():
    sucCount = 0
    for pos in range(100):
        print(f"## Puzzle {pos} ##")
        success, resCn = fc_test_on_puzzle(pos)
        print(f"Success: {success}")
        if success:
            sucCount += 1
            print("Solution verified: {}".format(
                np.array_equal(es.extract_solutionArray_from_coreDict(resCn.coresDict), re.get_solution_array(pos))))
    print("#######")
    print(f"Summary: {sucCount} of 100 solved.")
    return sucCount


if __name__ == "__main__":
    full_run()
