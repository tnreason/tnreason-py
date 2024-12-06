from tnreason import engine

from tnreason.encoding import connectives as con
from tnreason.encoding import formulas_to_cores as enform
from tnreason.encoding import suffixes as suf


def parse_neuronNameDict_to_neuronColorDict(neuronNameDict):
    return {neuronName: [neuronNameDict[neuronName][0]] + [
        convert_candidateNames_to_colorList(candidatesList, neuronNameDict.keys()) for candidatesList in
        neuronNameDict[neuronName][1:]] for neuronName in neuronNameDict}


def convert_candidateNames_to_colorList(candidatesList, neuronNames):
    if isinstance(candidatesList, str):
        varKey, pos = candidatesList.split("=")
        return varKey + suf.categoricalVariableSuffix + "=" + pos
    else:
        colorList = []
        for candidate in candidatesList:
            if candidate in neuronNames:
                colorList.append(candidate + suf.neurVariableSuffix)
            else:
                colorList.append(candidate + suf.atomicVariableSuffix)
        return colorList


def create_architecture(neuronColorDict, headNeuronNames=[], coreType=None):
    """
    Creates a tensor network of neuron cores with selection colors
        * neuronDict: Dictionary specifying to each neuronName a list of candidates (for the connective and the arguments)
        * headNeurons: List of neuronNames to be associated with hard headCores
    """
    architectureCores = {}
    for neuronName in neuronColorDict.keys():
        architectureCores = {**architectureCores,
                             **create_neuron(neuronName, neuronColorDict[neuronName][0], {
                                 neuronName + suf.varSelPosPrefix + str(i): posCandidates for i, posCandidates in
                                 enumerate(neuronColorDict[neuronName][1:])
                             }, coreType=coreType)}
    for headNeuronName in headNeuronNames:
        architectureCores = {**architectureCores,
                             **enform.create_boolean_head(color=headNeuronName + suf.neurVariableSuffix,
                                                          headType="truthEvaluation")}
    return architectureCores


def create_neuron(neuronName, connectiveList, candidatesDict={}, coreType=None):
    """
    Creates the cores to one neuron 
        * neuronName: String to use as prefix of the key to each core
        * connectiveList: List of connectives to be selected
        * candidatesDict: Dictionary of lists of candidates to each argument of the neuron
    """
    neuronCores = {
        neuronName + suf.actSelCoreSuffix: create_connective_selectors(neuronName, candidatesDict.keys(),
                                                                       connectiveList, coreType=coreType)}
    for candidateKey in candidatesDict:
        neuronCores = {**neuronCores, **create_variable_selectors(
            candidateKey, candidatesDict[candidateKey], coreType=coreType)}
    return neuronCores


def create_variable_selectors(candidateKey, variables,
                              coreType=None):  # candidateKey was created by neuronName + p + str(pos)
    """
    Creates the selection cores to one argument at a neuron.
    There are two possibilities to specify variables
        * list of variables string: Representing a selection of atomic variables represented in the string and a CP decomposition is created.
        * single string: Representing a categorical variable in the format X=[m] and a single selection core is created.
    Resulting colors in each core: [selection variable, candidate variable, neuron argument variable]
    """
    if isinstance(variables, str):
        catName, dimBracket = variables.split("=")
        dim = int(dimBracket.split("[")[1][:-1])

        selFunc = lambda s, c: [c == s]  # Whether selection variable coincides with control variable
        return {candidateKey + "_" + variables + suf.varSelCoreSuffix: engine.create_relational_encoding(
            inshape=[dim, dim], outshape=[2], incolors=[candidateKey + suf.varSelVarSuffix, catName],
            outcolors=[candidateKey],
            function=selFunc, coreType=coreType,
            name=candidateKey + "_" + variables + suf.varSelCoreSuffix)}
    cSelectorDict = {}
    for i, variableKey in enumerate(variables):
        coreFunc = lambda c, a, o: (not (c == i)) or (a == o)
        cSelectorDict[candidateKey + "_" + variableKey + suf.varSelCoreSuffix] = engine.create_tensor_encoding(
            inshape=[len(variables), 2, 2], incolors=[candidateKey + suf.varSelVarSuffix, variableKey, candidateKey],
            function=coreFunc, coreType=coreType,
            name=candidateKey + "_" + variableKey + suf.varSelCoreSuffix
        )
    return cSelectorDict


def create_connective_selectors(neuronName, candidateKeys, connectiveList, coreType=None):
    """
    Creates the connective selection core, using the candidateKeys as color and arity specification
    """
    if len(candidateKeys) == 1:
        return engine.create_relational_encoding(inshape=[len(connectiveList), 2], outshape=[2],
                                                 incolors=[neuronName + suf.actSelVarSuffix, *candidateKeys],
                                                 outcolors=[neuronName + suf.neurVariableSuffix],
                                                 function=con.get_unary_connective_selector(connectiveList),
                                                 coreType=coreType,
                                                 name=neuronName + suf.actSelCoreSuffix)
    elif len(candidateKeys) == 2:
        return engine.create_relational_encoding(inshape=[len(connectiveList), 2, 2], outshape=[2],
                                                 incolors=[neuronName + suf.actSelVarSuffix, *candidateKeys],
                                                 outcolors=[neuronName + suf.neurVariableSuffix],
                                                 function=con.get_binary_connective_selector(connectiveList),
                                                 coreType=coreType,
                                                 name=neuronName + suf.actSelCoreSuffix)
    else:
        raise ValueError(
            "Number {} of candidates wrong in Neuron {} with connectives {}!".format(len(candidateKeys), neuronName,
                                                                                     connectiveList))


## Functions to identify solution expressions when candidates are selected
def create_solution_expression(neuronNameDict, selectionDict):
    """
    Replaces the candidates of neurons by solutions and returns the identified head neurons as formulas
        * neuronNameDict: Dictionary specifying the neurons
        * selectionDict: Dictionary selecting candidates (connective and position) to each selection variables at each neuron
    """
    fixedNeurons = fix_neurons(neuronNameDict, selectionDict)
    headNeurons = get_headKeys(fixedNeurons)
    if len(headNeurons) != 1:
        print("WARNING: Headneurons are {}.".format(headNeurons))
    return {headKey: replace_neuronnames(headKey, fixedNeurons) for headKey in headNeurons}


def fix_neurons(neuronDict, selectionDict):
    """
    Replaces the neurons with subexpressions refering to each other
    Works both for neuronNameDict and neuronColorDict, since not checking whether variables are refering to neurons
    """
    rawFormulas = {}
    for neuronName in neuronDict:
        rawFormulas[neuronName] = [neuronDict[neuronName][0][selectionDict[neuronName + suf.actSelVarSuffix]]] + [
            fix_selection(neuronDict[neuronName][i],
                          selectionDict[neuronName + suf.varSelPosPrefix + str(i - 1) + suf.varSelVarSuffix])
            for i in range(1, len(neuronDict[neuronName]))]
    return rawFormulas


def fix_selection(choices, position):
    """
    Materializes a choice, either from a categorical variable (when choices is str) or from a list of possibilities (when choices is a list of str)
    """
    if isinstance(choices, str):  # The case of a categorical variable
        return choices.split("=")[0] + "=" + str(position)
    else:  # The case of a list of possibilities
        return choices[position]


def get_headKeys(fixedNeurons):
    """
    Identifies the independent neurons as heads,
    Works both for neuronColorsDicts and neuronNameDicts (split turns colors and names into names)
    """
    headKeys = set(fixedNeurons.keys())
    for formulaKey in fixedNeurons:
        for inNeuron in fixedNeurons[formulaKey][1:]:
            if len(suf.neurVariableSuffix) > 0:  # checks, whether a neuron suffix is given
                if inNeuron.split(suf.neurVariableSuffix)[0] in headKeys:
                    headKeys.remove(inNeuron.split(suf.neurVariableSuffix)[0])
            else:
                if inNeuron in headKeys:
                    headKeys.remove(inNeuron)
    return headKeys


def replace_neuronnames(currentNeuronName, fixedNeuronDict):
    """
    Replaces the current neuron with the respective expression, after iterative replacement of depending fixed neurons
    Works both for neuronColorDicts and neuronNameDicts (split turns colors and names into names)
    """
    if len(suf.neurVariableSuffix) > 0:  # Then need to strip the neural variable suffix of to compare with the keys
        if currentNeuronName.split(suf.neurVariableSuffix)[0] not in fixedNeuronDict:
            return currentNeuronName  ## Then an atom
        currentNeuron = fixedNeuronDict[currentNeuronName.split(suf.neurVariableSuffix)[0]].copy()
    else:
        if currentNeuronName not in fixedNeuronDict:
            return currentNeuronName  ## Then an atom
        currentNeuron = fixedNeuronDict[currentNeuronName].copy()

    currentNeuron = [currentNeuron[0]] + [replace_neuronnames(currentNeuron[i], fixedNeuronDict) for i in
                                          range(1, len(currentNeuron))]
    return currentNeuron


## Auxiliary functions for knowledge identifying the atoms and the dimension of selection variables
def find_atom_colors(specDict):
    atoms = set()
    for neuronName in specDict.keys():
        for positionList in specDict[neuronName][1:]:
            if isinstance(positionList, list):
                atoms = atoms | set([atomName + suf.atomicVariableSuffix for atomName in positionList])
    return list(atoms)


def find_selection_dimDict(specDict):
    dimDict = {}
    for neuronName in specDict:
        dimDict.update({neuronName + suf.actSelVarSuffix: len(specDict[neuronName][0]),
                        **{neuronName + suf.varSelPosPrefix + str(i) + suf.varSelVarSuffix: len(candidates)
                           for i, candidates in enumerate(specDict[neuronName][1:])}})
    return dimDict


def find_selection_colors(specDict):
    """
    Extracts the default selection colors from a architecture dict
    """
    colors = []
    for neuronName in specDict:
        colors.append(neuronName + suf.actSelVarSuffix)
        colors = colors + [neuronName + suf.varSelPosPrefix + str(i) + suf.varSelVarSuffix for i in
                           range(len(specDict[neuronName][1:]))]
    return colors
