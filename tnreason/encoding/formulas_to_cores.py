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
                                                    name=formulaName + suf.actCoreSuf),
                              **create_raw_formula_cores(expressionsDict[formulaName][:-1],
                                                         alreadyCreated=
                                                         list(knowledgeCores.keys()) + alreadyCreated,
                                                         coreType=coreType)}
        else:
            knowledgeCores = {**knowledgeCores,
                              **create_boolean_head(get_formula_color(expressionsDict[formulaName]), "truthEvaluation",
                                                    coreType=coreType, name=formulaName + suf.actCoreSuf),
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
    if get_formula_string(expression) + suf.comCoreSuf in alreadyCreated:
        return {}
    if isinstance(expression, str):
        return {}
    elif len(expression) == 1:
        assert isinstance(expression[0], str)
        return {}
    formulaCores = create_connective_core(expression, coreType=coreType)
    for subExpression in expression[1:]:
        formulaCores.update(create_raw_formula_cores(subExpression, alreadyCreated=alreadyCreated, coreType=coreType))
    return formulaCores


def create_connective_core(expression, coreType=None):
    """
    Creates the connective core at the head of the expression by loading the truth table
        * expression (script language)
    """
    if isinstance(expression, str):
        return {}
    return {get_formula_string(expression) + suf.comCoreSuf:
                engine.create_relational_encoding(inshape=[2 for _ in range(1, len(expression))], outshape=[2],
                                                  incolors=[get_formula_color(expression[i]) for i in
                                                            range(1, len(expression))],
                                                  outcolors=[get_formula_color(expression)],
                                                  function=con.get_connectives(expression[0]),
                                                  coreType=coreType,
                                                  name=get_formula_string(expression) + suf.comCoreSuf)}


def create_boolean_head(color, headType, weight=None, coreType=None, name=None):
    """
    Created the head core to a boolean variable (with dimension 2)
    CHANGE: Expfactor and uncertainty not boolean!
    """
    if headType == "truthEvaluation":
        headFunction = lambda x: x
    elif headType == "falseEvaluation":
        headFunction = lambda x: 1 - x
    elif headType == "expFactor":
        headFunction = lambda x: math.exp(weight * x)
    elif headType == "uncertaintyAsWeight":
        headFunction = lambda x: (1 - x) * (1 - weight) + x * weight
    else:
        raise ValueError("Headtype {} not understood!".format(headType))
    if name is None:
        name = color + suf.actCoreSuf
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
                                                name=color + suf.eviCorePre + suf.actCoreSuf,
                                                coreType=coreType))
        else:
            coreDict.update(create_boolean_head(color, headType="falseEvaluation",
                                                name=color + suf.eviCorePre + suf.actCoreSuf,
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
    Identifies a color with an expression by adding a suffix to distinguish distributed and computed variables
        * expression (script language)
    """
    formula_string = get_formula_string(expression)
    if isinstance(expression, str) or (isinstance(expression, list) and len(expression) == 1):
        # Case of distributed variable
        return formula_string + suf.disVarSuf
    else:  # Case of computed variable
        return formula_string + suf.comVarSuf


def get_formula_string(expression):
    """
    Identifies color to expression except suffix
        * expression (script language)
    """
    if isinstance(expression, str):  ## Expression is atomic
        return expression
    elif len(expression) == 1:  ## Expression is atomic, but provided in nested form
        assert isinstance(expression[0], str), "Not string: {} of len {}".format(expression[0])
        return expression[0]
    else:
        if not isinstance(expression[0], str):
            raise ValueError("Connective {} has wrong type!".format(expression[0]))
        return "(" + expression[0] + "_" + "_".join(
            [get_formula_string(entry) for entry in expression[1:]]) + ")"


def get_all_atoms(expressionsDict):
    """
    Identifies the leafs of the expressions in the expressionsDict as atoms, 
        * expressionsDict: In script language
    Output: Colors of atoms
    """
    atoms = set()
    for key in expressionsDict:
        atoms = atoms | get_atoms(expressionsDict[key])
    return list(atoms)


def get_atoms(expression):
    if isinstance(expression, str):  ## Then an atom
        return {expression + suf.disVarSuf}
    elif len(expression) == 1:  ## Then an atomic formula
        return {expression[0] + suf.disVarSuf}
    else:  ## Then a formula with connective in first position
        atoms = set()
        for subExpression in expression[1:]:
            atoms = atoms | get_atoms(subExpression)
        return atoms
