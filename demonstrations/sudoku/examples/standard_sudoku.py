from demonstrations.sudoku.constraints.sudoku_constraints import get_sudoku_constraints
from tnreason import representation, application

def get_assignment_as_CANetwork(num, startAssignment):
    """
    Prepares the Sudoku game with a start assignment as a Computation-Activation Network
    """
    constraints = get_sudoku_constraints(num)

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