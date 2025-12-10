from tnreason import engine


def get_sudoku_rules(num=3):
    """
    Creates a dictionary over the \woneoplus formulas (see Example 17)
    keys: decomposition variables,
    values: list of affected atomic variables
    """
    return {
        **{"I_:_:_" + str(c1) + "_" + str(c2) + "_" + str(n): [
            "X_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n) for r1 in range(num) for r2 in
            range(num)] for c1 in range(num) for c2 in range(num) for n in range(num ** 2)},  ## Columns
        **{"I_" + str(r1) + "_" + str(r2) + "_:_:_" + str(n): [
            "X_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n) for c1 in range(num) for c2 in
            range(num)] for r1 in range(num) for r2 in range(num) for n in range(num ** 2)},  ## Rows
        **{"I_:_" + str(r1) + "_:_" + str(c1) + "_" + str(n): [
            "X_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n) for r2 in range(num) for c2 in
            range(num)] for r1 in range(num) for c1 in range(num) for n in range(num ** 2)},  ## Squares
        **{"I_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_:": [
            "X_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n) for n in range(num ** 2)]
            for r1 in range(num) for r2 in range(num) for c1 in range(num) for c2 in range(num)}  ## Positions
    }


def create_decomposition_matrices(decompositionVariable, atomList):
    """
    Creates a tensor network of n^2 \tau^k hypercores to one Sudoku rule (see Example 18)
    """
    return {
        decompositionVariable + "_" + atomKey: engine.create_from_slice_iterator(
            shape=[2, len(atomList)],
            colors=[atomKey, decompositionVariable],
            sliceIterator=[(1, {atomKey: 0}),
                           (-1, {atomKey: 0, decompositionVariable: i}),
                           (1, {atomKey: 1, decompositionVariable: i})],
        )
        for i, atomKey in enumerate(atomList)
    }


def create_sudoku_rule_tensor_network(num):
    """
    Creates a tensor network of the n^2 \tau^k hypercores to each Sudoku rule (see Example 18)
    """
    rulesSpecDict = get_sudoku_rules(num)
    cores = {}
    for decomKey in rulesSpecDict:
        cores.update(create_decomposition_matrices(decomKey, rulesSpecDict[decomKey]))
    return cores


def encode_trivial_extended_evidence(E, num):
    return {**{str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n) + "_eC":
                   engine.create_from_slice_iterator(shape=[2],
                                                     colors=["X_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(
                                                         c2) + "_" + str(n)],
                                                     sliceIterator=[(1, {
                                                         "X_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(
                                                             c2) + "_" + str(n): 1})])
               for r1, r2, c1, c2, n in E},
            **{str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n) + "_eC":
                   engine.create_from_slice_iterator(shape=[2],
                                                     colors=["X_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(
                                                         c2) + "_" + str(n)], sliceIterator=[(1, {})]) for r1 in
               range(num) for r2 in range(num) for c1 in range(num)
               for c2 in range(num) for n in range(num ** 2) if (r1, r2, c1, c2, n) not in E}
            }


testNum = 2
assert len(create_sudoku_rule_tensor_network(testNum)) == 4 * testNum ** 6

evidenceCores = encode_trivial_extended_evidence(
    [(0, 0, 0, 0, 1), (1, 1, 0, 1, 0)], num=2
)
assert len(evidenceCores) == testNum ** 6
assert evidenceCores["0_0_0_0_0_eC"].values[0] == 1
assert evidenceCores["0_0_0_0_0_eC"].values[1] == 1
assert evidenceCores["0_0_0_0_1_eC"].values[0] == 0
assert evidenceCores["0_0_0_0_1_eC"].values[1] == 1

from demonstrations.comp_act_nets.algorithms import propagation as cp

def extract_resulting_evidence(propagator, num):
    return [
        (r1, r2, c1, c2, n) for r1 in range(num) for r2 in range(num) for c1 in range(num) for c2 in range(num) for n in
        range(num ** 2)
        if engine.contract({
            "own": propagator.computationNetwork[str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(
                c2) + "_" + str(n) + "_eC"],
            **propagator.messageDict[str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(
                c2) + "_" + str(n) + "_eC"]},
            openColors=["X_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(
                c2) + "_" + str(n)])[{"X_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(
            c2) + "_" + str(n): 0}] == 0
    ]


import numpy as np

def tuples_to_array(evidence, num=2):
    array = np.zeros(shape=(num ** 2, num ** 2))
    for (r1, r2, c1, c2, n) in evidence:
        array[r1 * num + r2, c1 * num + c2] = n + 1
    return array

num = 2
evidence = [
    (0, 0, 0, 0, 0), (0, 0, 1, 0, 2), (0, 0, 1, 1, 1),
    (0, 1, 0, 1, 1),
    (1, 0, 1, 0, 3),
    (1, 1, 0, 0, 3), (1, 1, 0, 1, 2)
]

print(tuples_to_array(evidence))

propagator = cp.ContractionPropagation(
    tensorNetwork={**create_sudoku_rule_tensor_network(num=num), **encode_trivial_extended_evidence(evidence, num=num)})
propagator.constraint_propagation(
    startSendKeys=["_".join([str(i) for i in evidenceTuple]) + "_eC" for evidenceTuple in evidence])
solutionArray = tuples_to_array(extract_resulting_evidence(propagator, num=2))

print(solutionArray)

assert solutionArray[0, 0] == 1
assert solutionArray[0, 1] == 4
assert solutionArray[0, 2] == 3
assert solutionArray[0, 3] == 2
assert solutionArray[1, 0] == 3
assert solutionArray[1, 1] == 2
assert solutionArray[1, 2] == 1
assert solutionArray[1, 3] == 4
assert solutionArray[2, 0] == 2
assert solutionArray[2, 1] == 1
assert solutionArray[2, 2] == 4
assert solutionArray[2, 3] == 3
assert solutionArray[3, 0] == 4
assert solutionArray[3, 1] == 3
assert solutionArray[3, 2] == 2
assert solutionArray[3, 3] == 1
