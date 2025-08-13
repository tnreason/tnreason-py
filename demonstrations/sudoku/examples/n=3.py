from tnreason import reasoning

from demonstrations.sudoku import representation as rep
from demonstrations.sudoku import solution as sol
from demonstrations.sudoku import visualization as vis

num = 3

evidenceDict = {
    "a_0_1_0_0_1": 1,
    "a_0_0_0_1_0": 1,
    "a_0_1_1_0_3": 1,
    "a_1_0_0_0_2": 1,
    "a_0_0_1_0_2": 1
}
start_array = vis.evidence_to_array(evidenceDict, num=num)
vis.visualize_sudoku(start_array.astype(int), number=3, label="Start Assignment")

## Representation ##

caNetwork = rep.get_assignment_as_CANetwork(num=num, startAssignment=evidenceDict)

## Reasoning ##

propagator = reasoning.get_inferer("ExpectationPropagator")(
    caNetwork=caNetwork,
    clusterDict=sol.get_simple_clusterDict(num=num),
    meanParamDict=dict()
)
propagator.propagate_until_convergence(evidenceDict.keys(), maxMessageCount=300)

solution_array = vis.evidence_to_array(sol.meanParams_to_evidence(propagator.meanParamDict), num=num)
# Already Known
assert solution_array[0, 3] == 3
assert solution_array[1, 0] == 2
assert solution_array[1, 3] == 4
assert solution_array[3, 0] == 3
vis.visualize_sudoku(solution_array.astype(int), number=3, label="Solution Assignment")

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
# assert assignmentDict["square_1_0_0"] == num)
# assert not assignmentDict["a_0_0_0_0_1"])
