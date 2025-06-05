from tnreason import representation

from tnreason.application import connectives as con
from tnreason.representation import suffixes as suf

import math

## To be dropped: avoid create_boolean head!
def create_formulas_cores(expressionsDict, alreadyCreated=[], coreType=None):
    """
    Creates a tensor network of connective and head cores
        * expressionsDict (script language): Dictionary of nested listed representing expressions
        * alreadyCreated: List of keys to computation cores to be omitted
    """
    knowledgeCores = {}
    for formulaName in expressionsDict.keys():
        if isinstance(expressionsDict[formulaName][-1], float) or isinstance(expressionsDict[formulaName][-1], int):
            knowledgeCores = {**knowledgeCores,
                formulaName + suf.actCoreSuf : representation.create_tensor_encoding(inshape=[2], incolors=[get_formula_color(expressionsDict[formulaName][:-1])],
                                                                                     function= lambda x : math.exp(expressionsDict[formulaName][-1] * x),
                                                                                     coreType=coreType, name=formulaName + suf.actCoreSuf),
                              **create_formula_computation_cores(expressionsDict[formulaName][:-1],
                                                                 alreadyCreated=
                                                                 list(knowledgeCores.keys()) + alreadyCreated,
                                                                 coreType=coreType)}
        else:
            knowledgeCores = {**knowledgeCores,
                              formulaName + suf.actCoreSuf: representation.create_basis_core(
                                  name=formulaName + suf.actCoreSuf, shape=[2], colors=[
                                      get_formula_color(expressionsDict[formulaName])], numberTuple=(1),
                                  coreType=coreType),
                              **create_formula_computation_cores(expressionsDict[formulaName],
                                                                 alreadyCreated=list(
                                                                     knowledgeCores.keys()) + alreadyCreated,
                                                                 coreType=coreType)}
    return knowledgeCores


def create_expressionDict_computation_cores(expressionDict, coreType=None):
    """
    Creates the connective cores to an expression, omitting the elsewhere created cores
        * expressionDict: Dictionary of nested lists specifying formulas, possible with a canonical parameter in the end (dropped by drop_canParam())
        * alreadyCreated: List of keys to connective cores to be omitted
    """
    computationCores = dict()
    for expressionKey in expressionDict:
        computationCores.update(
            create_formula_computation_cores(drop_canParam(expressionDict[expressionKey]),
                                             alreadyCreated=list(computationCores.keys()), coreType=coreType))
    return computationCores


def drop_canParam(expression):
    if isinstance(expression[-1], float) or isinstance(expression[-1], int):
        return expression[:-1]
    else:
        return expression


def create_formula_computation_cores(expression, alreadyCreated=[], coreType=None):
    """
    Creates the connective cores to an expression, omitting the elsewhere created cores
        * expression: Nested list specifying a formula, possible with a canonical parameter in the end
        * alreadyCreated: List of keys to connective cores to be omitted
    """
    if get_formula_string(
            expression) + suf.comCoreSuf in alreadyCreated:  # Then redundant, and all subexpressions are not created
        return dict()
    elif isinstance(expression, str):  # Case of atomic fact
        return dict()
    elif len(expression) == 1:  # Case of atomic fact
        assert isinstance(expression[0], str)
        return dict()
    else:
        formulaCores = create_connective_core(expression, coreType=coreType)
        for subExpression in expression[1:]:
            formulaCores.update(
                create_formula_computation_cores(subExpression, alreadyCreated=alreadyCreated, coreType=coreType))
        return formulaCores


def create_connective_core(expression, coreType=None):
    """
    Creates the connective core at the head of the expression by loading the truth table
        * expression (script language)
    """
    if isinstance(expression, str):
        return dict()
    if isinstance(expression[-1], float) or isinstance(expression[-1], int):
        cardinality = len(expression) - 2  # Since the first is the connective and the last is the canonical parameter
    else:
        cardinality = len(expression) - 1  # Since the first is the connective

    return {get_formula_string(expression) + suf.comCoreSuf:
        representation.create_relational_encoding(
            inshape=[2 for _ in range(1, cardinality + 1)], outshape=[2],
            incolors=[get_formula_color(expression[i]) for i in
                      range(1, cardinality + 1)],
            outcolors=[get_formula_color(expression)],
            indicesToIndicesFunction=con.get_connectives(expression[0]),
            coreType=coreType,
            name=get_formula_string(expression) + suf.comCoreSuf)}


def create_evidence_cores(evidenceDict, coreType=None):
    coreDict = dict()
    for color in evidenceDict:
        if evidenceDict[color]:
            coreDict[color + suf.eviCoreIn + suf.actCoreSuf] = representation.create_basis_core(
                name=color + suf.eviCoreIn + suf.actCoreSuf,
                shape=[2],
                colors=[color],
                numberTuple=(1), coreType=coreType)
        else:
            coreDict[color + suf.eviCoreIn + suf.actCoreSuf] = representation.create_basis_core(
                name=color + suf.eviCoreIn + suf.actCoreSuf,
                shape=[2],
                colors=[color],
                numberTuple=(0), coreType=coreType)
    return coreDict


def create_atom_evidence_cores(evidenceDict, coreType=None):
    """
    Turns positive and negative evidence about atoms into literal formulas and encodes them as facts
    """
    return create_evidence_cores({get_formula_color(atomKey): evidenceDict[atomKey] for atomKey in evidenceDict},
                                 coreType=coreType)


def get_formula_color(expression):
    """
    Identifies a color with an expression by adding a suffix to distinguish distributed and computed variables
        * expression (script language) possibly with canParam on last position
    """
    formula_string = get_formula_string(drop_canParam(expression))
    if isinstance(expression, str) or (isinstance(expression, list) and len(expression) == 1):
        # Case of distributed variable
        return formula_string + suf.disVarSuf
    else:  # Case of computed variable
        return formula_string + suf.comVarSuf


def get_formula_string(expression):
    """
    Identifies color to expression except suffix
        * expression (script language) without canParam (use the drop_canParam before)
    """
    if isinstance(expression, str):  ## Expression is atomic
        return expression
    elif len(expression) == 1:  ## Expression is atomic, but provided in nested form
        assert isinstance(expression[0], str), "Failed to understood as atom: {}".format(expression[0])
        return expression[0]
    else:
        if not isinstance(expression[0], str):
            raise ValueError("Connective {} has wrong type!".format(expression[0]))
        return "(" + expression[0] + "_" + "_".join(
            [get_formula_string(entry) for entry in expression[1:]]) + ")"
