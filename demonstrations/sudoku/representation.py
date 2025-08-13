from tnreason import representation, application

"""
We here provide the main Sudoku constraints by dictionaries of categorical constraints.
These constraints can be represented by tensor networks using tnreason.representation.create_categorical_cores(constraints).
"""

def get_assignment_as_CANetwork(num, startAssignment):
    """
    Prepares the Sudoku game with a start assignment as a Computation-Activation Network
    """
    constraints = get_sudoku_constraints(num)

    atomVariables = ["a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n)
                     for r1 in range(num) for r2 in range(num) for c1 in range(num) for c2 in range(num) for n in
                     range(num ** 2)]

    return representation.ComputationActivationNetwork(
        featureDict={
            **{atomKey: representation.HardPartitionFeature(featureColors=[atomKey], affectedComputationCores={}) for
               atomKey in atomVariables},  # Atomic Variable Features
            **{categoricalKey: representation.HardPartitionFeature(featureColors=[categoricalKey],
                                                                   affectedComputationCores={},
                                                                   shape=[num ** 2])
               for categoricalKey in constraints},
            **{categoricalKey + "_" + atomKey: representation.HardPartitionFeature(
                featureColors=[categoricalKey, atomKey],
                shape=[num ** 2, 2],
                affectedComputationCores=[
                    categoricalKey + "_" + atomKey + representation.suf.atoCoreSuf])
               for categoricalKey in constraints for atomKey in constraints[categoricalKey]}
        },
        computationCoreDict=application.create_categorical_cores(constraints),
        canParamDict={evidenceKey: representation.create_basis_core(name=evidenceKey, shape=[2],
                                                                    colors=[evidenceKey],
                                                                    numberTuple=[
                                                                                           startAssignment[evidenceKey]])
                      for evidenceKey in startAssignment}
    )

def get_sudoku_constraints(num=3):
    return {**get_column_constraints(num),
            **get_row_constraints(num),
            **get_squares_constraints(num),
            **get_position_constraints(num)}


def get_column_constraints(num=3):
    """
    In each column each number appears exactly once.
    """
    return {"col_" + str(n) + "_" + str(c1) + "_" + str(c2): [
        "a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n) for r1 in range(num) for r2 in
        range(num)] for c1 in range(num) for c2 in range(num) for n in range(num ** 2)}


def get_row_constraints(num=3):
    """
    In each row each number appears exactly once.
    """
    return {"row_" + str(n) + "_" + str(r1) + "_" + str(r2): [
        "a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n) for c1 in range(num) for c2 in
        range(num)
    ] for r1 in range(num) for r2 in range(num) for n in
        range(num ** 2)}


def get_squares_constraints(num=3):
    """
    In each square each number appears exactly once.
    """
    return {"square_" + str(n) + "_" + str(r1) + "_" + str(c1): [
        "a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n)
        for r2 in range(num) for c2 in range(num)
    ]
        for r1 in range(num) for c1 in range(num) for n in range(num ** 2)}


def get_position_constraints(num=3):
    """
    Exactly one number is assigned at each position.
    """
    return {"pos_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2): [
        "a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n)
        for n in range(num ** 2)
    ] for r1 in range(num) for r2 in range(num) for c1 in range(num) for c2 in range(num)}


