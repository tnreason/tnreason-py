from tnreason import engine

defaultSparseCoreType = "PolynomialCore"


## Summarizing
def weightedFormulas_to_sparseCore(weightedFormulas, coreType=defaultSparseCoreType):
    """
    Builds the weighted sum of the formulas as a slice-sparse Core
    """
    sumCore = engine.get_core(coreType)(values=[], colors=[], shape=[])
    for key in weightedFormulas:
        sumCore = sumCore + (
                    weightedFormulas[key][-1] * formula_to_sparseCore(weightedFormulas[key][:-1], coreType=coreType))
    sumCore.add_identical_slices()
    return sumCore


def formula_to_sparseCore(expression, coreType=defaultSparseCoreType):
    """
    Turns an expression into a slice-sparse Core based on the contraction of its clauses
    """
    core = clauseList_to_core(simplify_clauseList(cnf_to_dict(to_cnf(expression, uppushAnd=False))), coreType=coreType)
    if coreType == "PolynomialCore":
        core.add_identical_slices()
    return core


## ClauseList to Cores
def clause_to_core(variablesDict, coreType=defaultSparseCoreType):
    return engine.create_from_slice_iterator(shape=[2 for vKey in variablesDict],
                                             colors=[vKey for vKey in variablesDict],
                                             sliceIterator=iter([(1, dict()), (
                                                 -1, {vKey: 1 - variablesDict[vKey] for vKey in variablesDict})]),
                                             coreType=coreType
                                             )


def clauseList_to_core(clauseList, coreType=defaultSparseCoreType, contractionMethod="CorewiseContractor"):
    """
    Provides a CP-sparsity-based way to encode formulas based on their cnf,
    Alternative to tensor network based in formula_to_cores
    """
    return engine.contract(
        coreDict={"c" + str(i): clause_to_core(variablesDict, coreType=coreType) for i, variablesDict in
                  enumerate(clauseList)},
        openColors=list(set.union(*[set(clauses.keys()) for clauses in clauseList])),
        contractionMethod=contractionMethod,
        coreType=coreType
    )


## CNF to ClauseList (Syntax Manipulation Only!)
def cnf_to_dict(expression, atomsOnly=True):
    # atomsOnly: Whether all keys in the dictionary refer to 2-dimensional categorical varibables (standard)
    if isinstance(expression, str):
        return [{expression: 1}]  ## Then a positive Literal
    elif len(expression) == 2:
        if expression[0] == "not":  ## Then a negative Literal
            return [{expression[1]: 0}]
        elif expression[0] == "id":
            return cnf_to_dict(expression[1])
    elif len(expression) == 3:
        if expression[0] == "and":
            return [entry for entry in cnf_to_dict(expression[1]) if len(entry) != 0] + [entry for entry in
                                                                                         cnf_to_dict(expression[2]) if
                                                                                         len(entry) != 0]
        elif expression[0] == "or":
            combinedClauses = []
            for leftClause in cnf_to_dict(expression[1]):
                for rightClause in cnf_to_dict(expression[2]):
                    if atomsOnly and all([rightClause[key] == leftClause[key] for key in
                                          set(rightClause.keys()) & set(leftClause.keys())]):
                        combinedClauses.append({**leftClause, **rightClause})
            if len(combinedClauses) == 0:  ## If all clauses got trivial
                return [dict()]
            return combinedClauses


def simplify_clauseList(clauseList):
    simplifiedList = clauseList.copy()
    for clause in clauseList:
        simplifiedList.remove(clause)
        if not any([all([clause[key] == testClause[key] for key in set(clause.keys()) & set(testClause.keys())])
                    and set(testClause.keys()) <= clause.keys() for testClause in simplifiedList]):
            # Check whether any testClause in simplified list is contained in the clause
            simplifiedList.append(clause)
    return simplifiedList


## Expressions to CNF
def to_cnf(expression, uppushAnd=False):  ## Allowing for ors before ands if uppushAnd=False
    if not isinstance(expression, str) and len(expression) == 1:  # To handle stripped weightedFormulas
        expression = expression[0]
    expression = eliminate_eq_xor(expression)
    expression = eliminate_imp(expression)
    expression = groundpush_not(expression)
    if uppushAnd:
        expression = uppush_and(expression)
    return expression


def eliminate_eq_xor(expression):
    if isinstance(expression, str):
        return expression
    elif len(expression) == 2:
        return [expression[0], eliminate_eq_xor(expression[1])]
    elif len(expression) == 3:
        if expression[0] == "eq":
            return ["and", ["imp", eliminate_eq_xor(expression[1]), eliminate_eq_xor(expression[2])],
                    ["imp", eliminate_eq_xor(expression[2]), eliminate_eq_xor(expression[1])]]
        elif expression[0] == "xor":
            return ["not", ["and", ["imp", eliminate_eq_xor(expression[1]), eliminate_eq_xor(expression[2])],
                            ["imp", eliminate_eq_xor(expression[2]), eliminate_eq_xor(expression[1])]]]
        else:
            return [expression[0], eliminate_eq_xor(expression[1]), eliminate_eq_xor(expression[2])]
    else:
        raise ValueError("Expression {} not understood!".format(expression))


def eliminate_imp(expression):
    if isinstance(expression, str):
        return expression
    elif len(expression) == 2:
        return [expression[0], eliminate_imp(expression[1])]
    elif len(expression) == 3:
        if expression[0] == "imp":
            return ["or", ["not", eliminate_imp(expression[1])], eliminate_imp(expression[2])]
        else:
            return [expression[0], eliminate_imp(expression[1]), eliminate_imp(expression[2])]
    else:
        raise ValueError("Expression {} not understood!".format(expression))


def groundpush_not(expression):
    if isinstance(expression, str):
        return expression
    elif len(expression) == 2:  # Then assume that connective is not
        if isinstance(expression[1], str):  ## Already at literal level
            return expression
        elif len(expression[1]) == 2:
            return groundpush_not(expression[1][1])  # Case of double not
        elif len(expression[1]) == 3:
            if expression[1][0] == "and":
                return ["or", groundpush_not(["not", expression[1][1]]), groundpush_not(["not", expression[1][2]])]
            elif expression[1][0] == "or":
                return ["and", groundpush_not(["not", expression[1][1]]), groundpush_not(["not", expression[1][2]])]
            else:
                raise ValueError("Expression {} not groundpushable!".format(expression))
    elif len(expression) == 3:
        return [expression[0], groundpush_not(expression[1]), groundpush_not(expression[2])]

    else:
        raise ValueError("Expression {} not groundpushable!".format(expression))


def uppush_and(
        expression):  ## Redundant, only when aiming at a CNF in nested language -> Better to go to clauseLists before that!
    while not and_above_or_checker(expression):
        expression = and_or_modify(expression)
    return expression


def and_or_modify(expression):
    if isinstance(expression, str):
        return expression
    elif len(expression) == 2:
        return [expression[0], and_or_modify(expression[1])]
    elif len(expression) == 3:
        if expression[0] == "or":
            if not_starts_with_and(expression[1]) and not not_starts_with_and(expression[2]):
                return ["and", ["or", expression[1], expression[2][1]], ["or", expression[1], expression[2][2]]]
            elif not not_starts_with_and(expression[1]) and not_starts_with_and(expression[2]):
                return ["and", ["or", expression[2], expression[1][1]], ["or", expression[2], expression[1][2]]]
            elif not not_starts_with_and(expression[1]) and not not_starts_with_and(expression[2]):
                return ["and",
                        ["and", ["or", expression[1][1], expression[2][1]], ["or", expression[1][1], expression[2][2]]],
                        ["and", ["or", expression[1][2], expression[2][1]], ["or", expression[1][2], expression[2][2]]]]
        return [expression[0], and_or_modify(expression[1]), and_or_modify(expression[2])]


def not_starts_with_and(expression):
    if isinstance(expression, str):
        return True
    else:
        return not expression[0] == "and"


def and_above_or_checker(expression):
    if isinstance(expression, str):
        return True
    elif len(expression) == 2:
        return True
    elif len(expression) == 3:
        if expression[0] == "or":
            return not_containing_and(expression)
        else:
            return and_above_or_checker(expression[1]) and and_above_or_checker(expression[2])


def not_containing_and(expression):
    if isinstance(expression, str):
        return True
    elif len(expression) == 2:
        return True
    elif len(expression) == 3:
        if expression[0] == "and":
            return False
        else:
            return not_containing_and(expression[1]) and not_containing_and(expression[2])
