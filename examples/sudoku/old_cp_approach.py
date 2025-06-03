from examples.sudoku import representation as rep
from examples.sudoku import visualization as vis

import numpy as np

from tnreason import reasoning
from tnreason import representation

num = 2
structureCores = representation.create_categorical_cores(rep.get_sudoku_constraints(num=num))
# preEvidence = {
#             "a_0_1_0_0_1": 1,
#             "a_0_0_0_1_0": 1,
#             "a_0_1_1_0_3": 1,
#             "a_1_0_0_0_2": 1,
#             "a_0_0_1_0_2": 1
#         }

sudoku_array = np.array([
    [3, 0, 0, 0],
    [4, 1, 3, 0],
    [2, 0, 0, 0],
    [1, 3, 0, 0],
])

def array_to_catEvidence(array, num, verbose=False):
    evidenceDict = {}
    rearranged = array.reshape((num, num, num, num))
    for r1 in range(num):
        for r2 in range(num):
            for c1 in range(num):
                for c2 in range(num):
                    if rearranged[r1, r2, c1, c2] != 0:
                        evidenceDict["pos_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2)] = rearranged[
                                                                                                             r1, r2, c1, c2] - 1
    return evidenceDict

def catEvidence_to_atomEvidence(catEvidence):
    return {"a_" + "_".join(key.split("_")[1:]) + "_" + str(catEvidence[key]): 1 for key in catEvidence}

catEvidence = array_to_catEvidence(sudoku_array, num = 2)
preEvidence = catEvidence_to_atomEvidence(catEvidence)

propagator = reasoning.ConstraintPropagator(
            {**structureCores,
             **representation.create_atom_evidence_cores(preEvidence)},
            verbose=False
        )
propagator.propagate_cores()
assignmentDict = propagator.find_assignments()

array = rep.evidenceDict_to_array(assignmentDict, 2)
print(array)

vis.visualize_sudoku(array, number=2) # Missing values, should be fully infered! Due to atom color clash?