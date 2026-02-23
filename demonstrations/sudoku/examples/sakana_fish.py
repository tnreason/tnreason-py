"""
We here provide templates for creating the cores needed for implementing the constraints on the Sakana Fish Sudoku puzzle.

We exploit either the more fundamental operations of tnreason.engine, or more abstract functionality from tnreason.representation.

To be more precise:
- Red line: We instantiate the hidden variable that indicates whether a position is odd or even, and we create the constraints by xor on the hidden variables.
- White dot: We create the constraint that two positions differ by one, based .
- Black dot: We create the constraint that one position is double the other.
"""

from collections import ChainMap

from tnreason import representation, reasoning, engine, application

from demonstrations.sudoku.constraints import sudoku_constraints as rep
from demonstrations.sudoku.constraints.white_dot import white_dot_constraint, cenc_white_dot_constraint
from demonstrations.sudoku.constraints.black_dot import black_dot_constraint, cenc_black_dot_constraint
from demonstrations.sudoku.constraints.red_line import red_line_constraint_from_odd, red_line_constraint, prepare_odd_indicator_core, benc_prepare_odd_indicator_core, cenc_red_line_constraint

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

def rc_to_pos_assignment(r,c):
    r1 = r // 3
    r2 = r % 3
    c1 = c // 3
    c2 = c % 3
    return "pos_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2)

def get_assignment_as_CANetwork(num, startAssignment):
    """
    Prepares the Sudoku game with a start assignment as a Computation-Activation Network
    """
    constraints_Sudoku = rep.get_sudoku_constraints(num)
    comCoreDict_Sudoku = application.create_categorical_cores(constraints_Sudoku)

    # white_dots = [("pos_" + str(0) + "_" + str(0) + "_" + str(0) + "_" + str(0),"pos_" + str(0) + "_" + str(1) + "_" + str(0) + "_" + str(0)),
    #               ("pos_" + str(0) + "_" + str(0) + "_" + str(0) + "_" + str(1),"pos_" + str(0) + "_" + str(0) + "_" + str(0) + "_" + str(2)),
    #               ("pos_" + str(0) + "_" + str(0) + "_" + str(1) + "_" + str(0),"pos_" + str(0) + "_" + str(1) + "_" + str(1) + "_" + str(0)),
    #               ("pos_" + str(0) + "_" + str(0) + "_" + str(1) + "_" + str(1),"pos_" + str(0) + "_" + str(1) + "_" + str(1) + "_" + str(1)),
    #               ("pos_" + str(0) + "_" + str(1) + "_" + str(2) + "_" + str(0),"pos_" + str(0) + "_" + str(2) + "_" + str(2) + "_" + str(0)),
    #               ("pos_" + str(1) + "_" + str(0) + "_" + str(2) + "_" + str(1),"pos_" + str(1) + "_" + str(0) + "_" + str(2) + "_" + str(2)),
    #               ]
    white_dots = [(rc_to_pos_assignment(0,0),rc_to_pos_assignment(1,0)),
                  (rc_to_pos_assignment(0,1),rc_to_pos_assignment(0,2)),
                  (rc_to_pos_assignment(0,3),rc_to_pos_assignment(1,3)),
                  (rc_to_pos_assignment(0,4),rc_to_pos_assignment(1,4)),
                  (rc_to_pos_assignment(1,6),rc_to_pos_assignment(2,6)),
                  (rc_to_pos_assignment(3,7),rc_to_pos_assignment(3,8)),
                  (rc_to_pos_assignment(4,0),rc_to_pos_assignment(4,1)),
                  (rc_to_pos_assignment(4,7),rc_to_pos_assignment(4,8)),
                  (rc_to_pos_assignment(5,7),rc_to_pos_assignment(5,8)),
                  (rc_to_pos_assignment(6,0),rc_to_pos_assignment(6,1)),
                  (rc_to_pos_assignment(6,0),rc_to_pos_assignment(7,0)),
                  (rc_to_pos_assignment(6,5),rc_to_pos_assignment(7,5)),
                  (rc_to_pos_assignment(6,6),rc_to_pos_assignment(6,7)),
                  (rc_to_pos_assignment(7,2),rc_to_pos_assignment(8,2)),
                  (rc_to_pos_assignment(7,4),rc_to_pos_assignment(7,5)),
                  (rc_to_pos_assignment(8,7),rc_to_pos_assignment(8,8))
                  ]
    comCoreDict_white_dot = {"white_dot_"+str(posVar1)+"_"+str(posVar2): cenc_white_dot_constraint(posVar1, posVar2, sudokuNum=num) for (posVar1, posVar2) in white_dots}

    black_dots = [(rc_to_pos_assignment(1,2),rc_to_pos_assignment(2,2)),
                  (rc_to_pos_assignment(2,1),rc_to_pos_assignment(3,1)),
                  (rc_to_pos_assignment(3,6),rc_to_pos_assignment(3,7)),
                  (rc_to_pos_assignment(7,7),rc_to_pos_assignment(8,7)),
                  (rc_to_pos_assignment(8,1),rc_to_pos_assignment(8,2))]
    comCoreDict_black_dot = {"black_dot_"+str(posVar1)+"_"+str(posVar2): cenc_black_dot_constraint(posVar1, posVar2, sudokuNum=num) for (posVar1, posVar2) in black_dots}

    red_line = [(rc_to_pos_assignment(2,1),rc_to_pos_assignment(2,2)),
                (rc_to_pos_assignment(2,2),rc_to_pos_assignment(2,3)),
                (rc_to_pos_assignment(2,3),rc_to_pos_assignment(2,4)),
                (rc_to_pos_assignment(2,4),rc_to_pos_assignment(2,5)),
                (rc_to_pos_assignment(2,5),rc_to_pos_assignment(3,6)),
                (rc_to_pos_assignment(3,6),rc_to_pos_assignment(4,7)),
                (rc_to_pos_assignment(4,7),rc_to_pos_assignment(5,6)),
                (rc_to_pos_assignment(6,5),rc_to_pos_assignment(6,4)),
                (rc_to_pos_assignment(6,4),rc_to_pos_assignment(6,3)),
                (rc_to_pos_assignment(6,3),rc_to_pos_assignment(6,2)),
                (rc_to_pos_assignment(6,2),rc_to_pos_assignment(6,1)),

                (rc_to_pos_assignment(2,3),rc_to_pos_assignment(3,3)),
                (rc_to_pos_assignment(3,3),rc_to_pos_assignment(4,2)),
                (rc_to_pos_assignment(4,2),rc_to_pos_assignment(5,1)),
                (rc_to_pos_assignment(5,1),rc_to_pos_assignment(5,0)),

                (rc_to_pos_assignment(3,0),rc_to_pos_assignment(3,1)),
                (rc_to_pos_assignment(3,1),rc_to_pos_assignment(4,2)),
                (rc_to_pos_assignment(4,2),rc_to_pos_assignment(5,3)),
                (rc_to_pos_assignment(5,3),rc_to_pos_assignment(6,3)),
                ]
    comCoreDict_red_line = [red_line_constraint(posVar1, posVar2, sudokuNum=num) for (posVar1, posVar2) in red_line]
    comCoreDict_red_line = dict(ChainMap(*comCoreDict_red_line))

    comCoreDict = {**comCoreDict_Sudoku, **comCoreDict_white_dot, **comCoreDict_black_dot, **comCoreDict_red_line}
    
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

    assert red_line_constraint_from_odd("o1", "o2")[{"o2": 1, "o1": 0}] == 1
    assert red_line_constraint_from_odd("o1", "o2")[{"o2": 1, "o1": 1}] == 0
    assert cenc_red_line_constraint("o1", "o2")[{"o2": 1, "o1": 0}] == 1
    assert cenc_red_line_constraint("o1", "o2")[{"o2": 1, "o1": 1}] == 0
    assert red_line_constraint("pos1", "pos2")["red_line_pos1_pos2"][{"oddInd_pos1": 0, "oddInd_pos2": 1}] == 1

    assert white_dot_constraint("p1", "p2")[{"p1": 1, "p2": 0}] == 1
    assert white_dot_constraint("p1", "p2")[{"p1": 5, "p2": 6}] == 1
    assert cenc_white_dot_constraint("p1", "p2")[{"p1": 1, "p2": 0}] == 1
    assert cenc_white_dot_constraint("p1", "p2")[{"p1": 5, "p2": 6}] == 1

    assert black_dot_constraint("p1", "p2")[{"p1": 5, "p2": 6}] == 0
    assert black_dot_constraint("p1", "p2")[{"p1": 2, "p2": 5}] == 1
    assert cenc_black_dot_constraint("p1", "p2")[{"p1": 5, "p2": 6}] == 0
    assert cenc_black_dot_constraint("p1", "p2")[{"p1": 2, "p2": 5}] == 1
    
    get_assignment_as_CANetwork(num=3, startAssignment={})
    solve_Sakana_Fish_Sudoku(num=3, startAssignment={})
