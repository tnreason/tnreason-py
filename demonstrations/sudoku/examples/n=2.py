from tnreason import reasoning

from demonstrations.sudoku import sudoku_constraints as rep
from demonstrations.sudoku import sudoku_forward_mp as sol
from demonstrations.sudoku import visualization as vis

num = 2

## Initialize a Sudoku Problem ##

evidenceDict = {
    "a_0_1_0_0_1": 1,
    "a_0_0_0_1_0": 1,
    "a_0_1_1_0_3": 1,
    "a_1_0_0_0_2": 1,
    "a_0_0_1_0_2": 1
}
start_array = vis.evidence_to_array(evidenceDict, num=num)
vis.visualize_sudoku(start_array.astype(int), number=2, label="Start Assignment")

## Representation ##

caNetwork = rep.get_assignment_as_CANetwork(num=num, startAssignment=evidenceDict)

## Reasoning ##

propagator = sol.get_forward_mp_inferer(num, evidenceDict)
propagator.propagate_until_convergence(evidenceDict.keys(),verbose=True)

## Investigate the Solution Assignment

solution_array = vis.evidence_to_array(sol.meanParams_to_evidence(propagator.meanParamDict), num=num)
assert solution_array[0, 0] == 4
assert solution_array[0, 1] == 1
assert solution_array[1, 0] == 2
assert solution_array[1, 1] == 3
assert solution_array[0, 2] == 3
assert solution_array[0, 3] == 2
assert solution_array[1, 2] == 4

vis.visualize_sudoku(solution_array.astype(int), number=2, label="Solution Assignment")
