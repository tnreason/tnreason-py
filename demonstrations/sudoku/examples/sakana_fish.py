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
    white_dots = [((0,0),(1,0))
                  ((0,1),(0,2)),
                  ((0,3),(1,3)),
                  ((0,4),(1,4)),
                  ((1,6),(2,6)),
                  ((3,7),(3,8)),
                  ((4,0),(4,1)),
                  ((4,7),(4,8)),
                  ((5,7),(5,8)),
                  ((6,0),(6,1)),
                  ((6,0),(7,0)),
                  ((6,5),(7,5)),
                  ((6,6),(6,7)),
                  ((7,2),(8,2)),
                  ((7,4),(7,5)),
                  ((8,7),(8,8))
                  ]
    white_dots = [(rc_to_pos_assignment(r1,c1),rc_to_pos_assignment(r2,c2)) for ((r1,c1),(r2,c2)) in white_dots]
    comCoreDict_white_dot = {"white_dot_"+str(posVar1)+"_"+str(posVar2): cenc_white_dot_constraint(posVar1, posVar2, sudokuNum=num) for (posVar1, posVar2) in white_dots}

    black_dots = [((1,2),(2,2)),
                  ((2,1),(3,1)),
                  ((3,6),(3,7)),
                  ((7,7),(8,7)),
                  ((8,1),(8,2))]
    black_dots = [(rc_to_pos_assignment(r1,c1),rc_to_pos_assignment(r2,c2)) for ((r1,c1),(r2,c2)) in black_dots]
    comCoreDict_black_dot = {"black_dot_"+str(posVar1)+"_"+str(posVar2): cenc_black_dot_constraint(posVar1, posVar2, sudokuNum=num) for (posVar1, posVar2) in black_dots}

    red_line = [((2,1),(2,2)),
                ((2,2),(2,3)),
                ((2,3),(2,4)),
                ((2,4),(2,5)),
                ((2,5),(3,6)),
                ((3,6),(4,7)),
                ((4,7),(5,6)),
                ((6,5),(6,4)),
                ((6,4),(6,3)),
                ((6,3),(6,2)),
                ((6,2),(6,1)),

                ((2,3),(3,3)),
                ((3,3),(4,2)),
                ((4,2),(5,1)),
                ((5,1),(5,0)),

                ((3,0),(3,1)),
                ((3,1),(4,2)),
                ((4,2),(5,3)),
                ((5,3),(6,3)),
                ]
    red_line = [(rc_to_pos_assignment(r1,c1),rc_to_pos_assignment(r2,c2)) for ((r1,c1),(r2,c2)) in red_line]
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
    
    get_assignment_as_CANetwork(num=3, startAssignment={})
    solve_Sakana_Fish_Sudoku(num=3, startAssignment={})
