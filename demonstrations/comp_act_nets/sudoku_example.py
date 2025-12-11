from tnreason.engine import contract
from tnreason.engine import create_from_slice_iterator as create


def create_sudoku_rule_tensor_network(num):
    """
    Creates a tensor network of n^2 \tau^k matrices to each Sudoku constraint
    """
    rulesSpecDict = {
        ## Column Constraints
        **{f"I_:_:_{c1}_{c2}_{n}": [f"X_{r1}_{r2}_{c1}_{c2}_{n}" for r1 in range(num) for r2 in
                                    range(num)] for c1 in range(num) for c2 in range(num)
           for n in range(num ** 2)},
        ## Row Constraints
        **{f"I_{r1}_{r2}_:_:_{n}": [f"X_{r1}_{r2}_{c1}_{c2}_{n}" for c1 in range(num) for c2 in
                                    range(num)] for r1 in range(num) for r2 in range(num)
           for n in range(num ** 2)},
        ## Squares Constraints
        **{f"I_{r1}_:_{c1}_:_{n}": [f"X_{r1}_{r2}_{c1}_{c2}_{n}" for r2 in range(num) for c2 in
                                    range(num)] for r1 in range(num) for c1 in range(num)
           for n in range(num ** 2)},
        ## Position Constraints
        **{f"I_{r1}_{r2}_{c1}_{c2}_:": [f"X_{r1}_{r2}_{c1}_{c2}_{n}" for n in range(num ** 2)]
           for r1 in range(num) for r2 in range(num) for c1 in range(num) for c2 in range(num)}
    }
    cores = {}
    for decomKey in rulesSpecDict:
        cores.update({
            decomKey + "_" + atomVar: create(
                shape=[2, len(rulesSpecDict[decomKey])],
                colors=[atomVar, decomKey],
                sliceIterator=[(1, {atomVar: 0}),
                               (-1, {atomVar: 0, decomKey: i}),
                               (1, {atomVar: 1, decomKey: i})])
            for i, atomVar in enumerate(rulesSpecDict[decomKey])
        })
    return cores


def encode_trivial_extended_evidence(E, num):
    return {**{f"{r1}_{r2}_{c1}_{c2}_{n}_eC":
                   create(shape=[2], colors=[f"X_{r1}_{r2}_{c1}_{c2}_{n}"],
                          sliceIterator=[(1, {f"X_{r1}_{r2}_{c1}_{c2}_{n}": 1})])
               for r1, r2, c1, c2, n in E},
            **{f"{r1}_{r2}_{c1}_{c2}_{n}_eC":
                   create(shape=[2], colors=[f"X_{r1}_{r2}_{c1}_{c2}_{n}"],
                          sliceIterator=[(1, {})])
               for r1 in range(num) for r2 in range(num) for c1 in range(num)
               for c2 in range(num) for n in range(num ** 2) if (r1, r2, c1, c2, n) not in E}}


def extract_resulting_evidence(propagator, num):
    return [
        (r1, r2, c1, c2, n) for r1 in range(num) for r2 in range(num) for c1 in range(num) for c2 in range(num) for
        n in range(num ** 2)
        if contract({
            "eC": propagator.cores[f"{r1}_{r2}_{c1}_{c2}_{n}_eC"],
            **propagator.messages[f"{r1}_{r2}_{c1}_{c2}_{n}_eC"]},
            openColors=[f"X_{r1}_{r2}_{c1}_{c2}_{n}"])[{f"X_{r1}_{r2}_{c1}_{c2}_{n}": 0}] == 0]


def tuples_to_array(evidence, num=2):
    array = np.zeros(shape=(num ** 2, num ** 2))
    for (r1, r2, c1, c2, n) in evidence:
        array[r1 * num + r2, c1 * num + c2] = n + 1
    return array


from demonstrations.comp_act_nets.algorithms import propagation as cp

num = 2
evidence = [(0, 0, 0, 0, 0), (0, 0, 1, 0, 2), (0, 0, 1, 1, 1),
            (0, 1, 0, 1, 1), (1, 0, 1, 0, 3), (1, 1, 0, 0, 3),
            (1, 1, 0, 1, 2)]

propagator = cp.ContractionPropagation(
    cores={**create_sudoku_rule_tensor_network(num=num),
           **encode_trivial_extended_evidence(evidence, num=num)})
propagator.constraint_propagation([f"{r1}_{r2}_{c1}_{c2}_{n}_eC" for (r1, r2, c1, c2, n) in evidence])
solutionArray = tuples_to_array(extract_resulting_evidence(propagator, num=2))
assert np.all(solutionArray == np.array([[1, 4, 3, 2], [3, 2, 1, 4], [2, 1, 4, 3], [4, 3, 2, 1]]))
