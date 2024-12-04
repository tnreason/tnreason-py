from tnreason import engine

from tnreason.encoding import connectives as con
from tnreason.encoding import suffixes as suf

import math


def create_formulas_cores(expressionsDict, alreadyCreated=[], coreType=None):
    """
    Creates a tensor network of connective and head cores
        * expressionsDict: Dictionary of nested listed representing expressions
        * alreadyCreated: List of keys to connective cores to be omitted
    """
    knowledgeCores = {}
    for formulaName in expressionsDict.keys():
        if isinstance(expressionsDict[formulaName][-1], float) or isinstance(expressionsDict[formulaName][-1], int):
            knowledgeCores = {**knowledgeCores,
                              **create_boolean_head(get_formula_color(expressionsDict[formulaName][:-1]), "expFactor",
                                                    weight=
                                                    expressionsDict[formulaName][-1], coreType=coreType,
                                                    name=formulaName + suf.headCoreSuffix),
                              **create_raw_formula_cores(expressionsDict[formulaName][:-1],
                                                         alreadyCreated=
                                                         list(knowledgeCores.keys()) + alreadyCreated,
                                                         coreType=coreType)}
        else:
            knowledgeCores = {**knowledgeCores,
                              **create_boolean_head(get_formula_color(expressionsDict[formulaName]), "truthEvaluation",
                                                    coreType=coreType, name=formulaName + suf.headCoreSuffix),
                              **create_raw_formula_cores(expressionsDict[formulaName],
                                                         alreadyCreated=list(knowledgeCores.keys()) + alreadyCreated,
                                                         coreType=coreType)}
    return knowledgeCores


def create_raw_formula_cores(expression, alreadyCreated=[], coreType=None):
    """
    Creates the connective cores to an expression, omitting the elsewhere created cores
        * expression: Nested list specifying a formula
        * alreadyCreated: List of keys to connective cores to be omitted
    """
    if get_formula_string(expression) + suf.connectiveCoreSuffix in alreadyCreated:
        return {}
    if isinstance(expression, str):
        return {}
    elif len(expression) == 1:
        assert isinstance(expression[0], str)
        return {}

    elif len(expression) == 2:
        return {**create_connective_core(expression, coreType=coreType),
                **create_raw_formula_cores(expression[1], alreadyCreated=alreadyCreated, coreType=coreType)}
    elif len(expression) == 3:
        return {**create_connective_core(expression, coreType=coreType),
                **create_raw_formula_cores(expression[1], alreadyCreated=alreadyCreated, coreType=coreType),
                **create_raw_formula_cores(expression[2], alreadyCreated=alreadyCreated, coreType=coreType)
                }
    else:
        raise ValueError("Expression {} not understood!".format(expression))


def create_connective_core(expression, coreType=None):
    """
    Creates the connective core at the head of the expression by loading the truth table
    """
    if isinstance(expression, str):
        return {}

    elif len(expression) == 2:
        return {get_formula_string(expression) + suf.connectiveCoreSuffix:
                    engine.create_relational_encoding(inshape=[2], outshape=[2],
                                                      incolors=[get_formula_color(expression[1])],
                                                      outcolors=[get_formula_color(expression)],
                                                      function=con.get_connectives(expression[0]),
                                                      coreType=coreType,
                                                      name=get_formula_string(expression) + suf.connectiveCoreSuffix)}

    elif len(expression) == 3:
        return {get_formula_string(expression) + suf.connectiveCoreSuffix:
                    engine.create_relational_encoding(inshape=[2, 2], outshape=[2],
                                                      incolors=[get_formula_color(expression[1]),
                                                                get_formula_color(expression[2])],
                                                      outcolors=[get_formula_color(expression)],
                                                      function=con.get_connectives(expression[0]),
                                                      coreType=coreType,
                                                      name=get_formula_string(expression) + suf.connectiveCoreSuffix)}
    else:
        raise ValueError("Expression {} not understood!".format(expression))


def create_boolean_head(color, headType, weight=None, coreType=None, name=None):
    """
    Created the head core to a boolean variable (with dimension 2)
    """
    if headType == "truthEvaluation":
        headFunction = lambda x: x
    elif headType == "falseEvaluation":
        headFunction = lambda x: 1 - x
    elif headType == "expFactor":
        headFunction = lambda x: math.exp(weight * x)
    else:
        raise ValueError("Headtype {} not understood!".format(headType))
    if name is None:
        name = color + suf.headCoreSuffix
    return {name: engine.create_tensor_encoding([2], [color], headFunction, coreType=coreType,
                                                name=name)}


def create_formula_head(expression, headType, weight=None, name=None, coreType=None):
    """
    Created the head core to an expression activating it, which is the boolean head to the formula color
    """
    return create_boolean_head(color=get_formula_color(expression), headType=headType, weight=weight, name=name,
                               coreType=coreType)


def create_evidence_cores(evidenceDict, coreType=None):
    coreDict = dict()
    for color in evidenceDict:
        if evidenceDict[color]:
            coreDict.update(create_boolean_head(color, headType="truthEvaluation",
                                                name=color + suf.evidenceCoreSuffix + suf.headCoreSuffix,
                                                coreType=coreType))
        else:
            coreDict.update(create_boolean_head(color, headType="falseEvaluation",
                                                name=color + suf.evidenceCoreSuffix + suf.headCoreSuffix,
                                                coreType=coreType))
    return coreDict


def create_atom_evidence_cores(evidenceDict, coreType=None):
    """
    Turns positive and negative evidence about atoms into literal formulas and encodes them as facts
    """
    return create_evidence_cores({get_formula_color(atomKey): evidenceDict[atomKey] for atomKey in evidenceDict},
                                 coreType=coreType)


def get_formula_color(expression):
    """
    Identifies a color with an expression
    """
    formula_string = get_formula_string(expression)
    if isinstance(expression, str) or (isinstance(expression, list) and len(expression) == 1):
        return formula_string + suf.atomicVariableSuffix
    else:
        return formula_string + suf.categoricalVariableSuffix

    # if isinstance(expression, str):  ## Expression is atomic
    #     return expression + suf.atomicVariableSuffix
    # elif len(expression) == 1:  ## Expression is atomic, but provided in nested form
    #     assert isinstance(expression[0], str)
    #     return expression[0] + suf.atomicVariableSuffix
    # else:
    #     if not isinstance(expression[0], str):
    #         raise ValueError("Connective {} has wrong type!".format(expression[0]))
    #     return "(" + expression[0] + "_" + "_".join(
    #         [get_formula_color(entry) for entry in expression[1:]]) + ")" + suf.categoricalVariableSuffix


def get_formula_string(expression):
    if isinstance(expression, str):  ## Expression is atomic
        return expression
    elif len(expression) == 1:  ## Expression is atomic, but provided in nested form
        assert isinstance(expression[0], str)
        return expression[0]
    else:
        if not isinstance(expression[0], str):
            raise ValueError("Connective {} has wrong type!".format(expression[0]))
        return "(" + expression[0] + "_" + "_".join(
            [get_formula_string(entry) for entry in expression[1:]]) + ")"


def get_all_atoms(expressionsDict):
    """
    Identifies the leafs of the expressions in the expressionsDict as atoms
    """
    atoms = set()
    for key in expressionsDict:
        atoms = atoms | get_atoms(expressionsDict[key])
    return list(atoms)


def get_atoms(expression):
    if isinstance(expression, str):  ## Then an atom
        return {expression+suf.atomicVariableSuffix}
    elif len(expression) == 1:  ## Then an atomic formula
        return {expression[0]+suf.atomicVariableSuffix}
    else:  ## Then a formula with connective in first position
        atoms = set()
        for subExpression in expression[1:]:
            atoms = atoms | get_atoms(subExpression)
        return atoms
