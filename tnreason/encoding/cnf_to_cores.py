from tnreason import engine


def clause_to_core(variablesDict, coreType="PolynomialCore"):
    return engine.create_from_slice_iterator(shape=[2 for vKey in variablesDict],
                                             colors=[vKey for vKey in variablesDict],
                                             sliceIterator=iter([(1, dict()), (
                                                 -1, {vKey: 1 - variablesDict[vKey] for vKey in variablesDict})]),
                                             coreType=coreType,
                                             name="ClauseCore"
                                             )


def clauseList_to_core(clauseList, coreType="PolynomialCore", contractionMethod="CorewiseContractor"):
    """
    Provides a CP-sparsity-based way to encode formulas based on their cnf,
    Alternative to tensor network based in formula_to_cores
    """
    return engine.contract(
        coreDict={"c" + str(i): clause_to_core(variablesDict, coreType=coreType) for i, variablesDict in
                  enumerate(clauseList)},
        openColors=list(set.union(*[set(clauses.keys()) for clauses in clauseList])),
        method=contractionMethod,
        coreType=coreType
    )
