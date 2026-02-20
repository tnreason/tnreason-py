"""
We here provide templates for creating the cores needed for implementing the constraints on the Sakana Fish Sudoku puzzle.

We exploit either the more fundamental operations of tnreason.engine, or more abstract functionality from tnreason.representation.

To be more precise:
- Red line: We instantiate the hidden variable that indicates whether a position is odd or even, and we create the constraints by xor on the hidden variables.
- White dot: We create the constraint that two positions differ by one, based .
- Black dot: We create the constraint that one position is double the other.
"""

from pathlib import Path
import json

from tnreason import representation, reasoning, engine, application

from demonstrations.sudoku.constraints import sudoku_constraints as rep
from demonstrations.sudoku.constraints.white_dot import white_dot_constraint, cenc_white_dot_constraint
from demonstrations.sudoku.constraints.black_dot import black_dot_constraint, cenc_black_dot_constraint
from demonstrations.sudoku.constraints.red_line import red_line_constraint, prepare_odd_indicator_core, benc_prepare_odd_indicator_core, cenc_prepare_odd_indicator_core, cenc_red_line_constraint

def get_forward_mp_inferer(num, startAssignment):
    constraints = rep.get_sudoku_constraints(num)
    atomVariables = ["a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n)
                     for r1 in range(num) for r2 in range(num) for c1 in range(num) for c2 in range(num) for n in
                     range(num ** 2)]

    caNetwork = get_assignment_as_CANetwork(num=num, startAssignment=startAssignment)
    messageClusters, inferenceClusters = reasoning.standard_clusters_from_computationCoreDict(
        caNetwork.computationCoreDict)

    return reasoning.ForwardMessagePasser(
        caNetwork = caNetwork,
        messageClusters=messageClusters,
        inferenceClusters = inferenceClusters,
        meanParamDict={
            **{atomKey: engine.create_from_slice_iterator(
                shape=[2], colors=[atomKey],
                sliceIterator=[(1, {})]) for atomKey in atomVariables},
            **{categoricalKey: engine.create_from_slice_iterator(
                shape=[num ** 2], colors=[categoricalKey],
                sliceIterator=[(1, {})]) for categoricalKey in constraints}},
        allowClearning=True  # aggressively drop settled features to help larger boards converge
    )


def get_assignment_as_CANetwork(num, startAssignment):
    """
    Prepares the Sudoku game with a start assignment as a Computation-Activation Network
    """
    constraints = rep.get_sudoku_constraints(num)
    
    # atomVariables = ["a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n)
    #                  for r1 in range(num) for r2 in range(num) for c1 in range(num) for c2 in range(num) for n in
    #                  range(num ** 2)]

    comCoreDict = application.create_categorical_cores(constraints)
    
    canParamDict = {evidenceKey: representation.create_basis_core(name=evidenceKey, shape=[2],
                                                                    colors=[evidenceKey],
                                                                    numberTuple=[startAssignment[evidenceKey]])
                      for evidenceKey in startAssignment}

    featureDict = representation.standard_featureDict_from_computationCoreDict(comCoreDict)
    
    caNet = representation.ComputationActivationNetwork(
        featureDict=featureDict,
        computationCoreDict=comCoreDict,
        canParamDict=canParamDict
    )
    return caNet

def solve_Sakana_Fish_Sudoku(num, startAssignment):
    propagator = get_forward_mp_inferer(num, startAssignment)
    propagator.propagate_until_convergence(startAssignment.keys(), maxMessageCount=30000)
    return propagator.meanParamDict

if __name__ == "__main__":
    assert prepare_odd_indicator_core("pos", "odd")[{"pos": 3, "odd": 0}] == 1
    assert prepare_odd_indicator_core("pos", "odd")[{"pos": 3, "odd": 1}] == 0
    assert benc_prepare_odd_indicator_core("pos", "odd")[{"pos": 3, "odd": 0}] == 1
    assert benc_prepare_odd_indicator_core("pos", "odd")[{"pos": 3, "odd": 1}] == 0

    assert red_line_constraint("o1", "o2")[{"o2": 1, "o1": 0}] == 1
    assert red_line_constraint("o1", "o2")[{"o2": 1, "o1": 1}] == 0
    assert cenc_red_line_constraint("o1", "o2")[{"o2": 1, "o1": 0}] == 1
    assert cenc_red_line_constraint("o1", "o2")[{"o2": 1, "o1": 1}] == 0

    assert white_dot_constraint("p1", "p2")[{"p1": 1, "p2": 0}] == 1
    assert white_dot_constraint("p1", "p2")[{"p1": 5, "p2": 6}] == 1
    assert cenc_white_dot_constraint("p1", "p2")[{"p1": 1, "p2": 0}] == 1
    assert cenc_white_dot_constraint("p1", "p2")[{"p1": 5, "p2": 6}] == 1

    assert black_dot_constraint("p1", "p2")[{"p1": 5, "p2": 6}] == 0
    assert black_dot_constraint("p1", "p2")[{"p1": 2, "p2": 5}] == 1
    assert cenc_black_dot_constraint("p1", "p2")[{"p1": 5, "p2": 6}] == 0
    assert cenc_black_dot_constraint("p1", "p2")[{"p1": 2, "p2": 5}] == 1

    solve_Sakana_Fish_Sudoku(num=3, startAssignment={})
