from tnreason import representation, application, engine

"""
We here provide the main Sudoku constraints by dictionaries of categorical constraints.
These constraints can be represented by tensor networks using tnreason.representation.create_categorical_cores(constraints).
"""


def get_clauseList_as_CANetwork(clauseList, evidenceDict=dict()):
    """
    Prepares the SAT instance as a Computation-Activation Network
    """
    atomVariables = list({key for clause in clauseList for key in clause.keys()})

    return representation.ComputationActivationNetwork(
        featureDict={
            **{"atomFeature_" + atomKey: representation.HardPartitionFeature(featureColors=[atomKey],
                                                                             affectedComputationCores={}) for
               atomKey in atomVariables},  # Atomic Variable Features
            **{"clauseFeature_" + str(i): representation.PassiveFeature(featureColors=list(clauseDict.keys()),
                                                                        affectedComputationCores={
                                                                            "clauseCore_" + str(i)},
                                                                        shape=[])
               for i, clauseDict in enumerate(clauseList)},
        },
        computationCoreDict={"clauseCore_" + str(i): create_clause_core(clauseDict) for i, clauseDict in
                             enumerate(clauseList)}
        ,
        canParamDict={evidenceKey: representation.create_basis_core(name=evidenceKey, shape=[2],
                                                                    colors=[evidenceKey],
                                                                    numberTuple=[evidenceDict[evidenceKey]])
                      for evidenceKey in evidenceDict}
    )


def create_clause_core(clauseDict, coreType=None):
    return engine.create_from_slice_iterator(colors=list(clauseDict.keys()), shape=[2] * len(clauseDict),
                                             sliceIterator=[(1, {}), (-1,
                                                                      {variableKey: int(1 - clauseDict[variableKey]) for
                                                                       variableKey in clauseDict})], coreType=coreType)


if __name__ == "__main__":
    clauseList = [{"a": 0, "b": 1}, {"c": 1}]
    caNetwork = get_clauseList_as_CANetwork(clauseList)
    print(caNetwork)
