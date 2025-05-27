from tnreason.representation import creation_handling as ch
from tnreason.representation import connectives as con
from tnreason.representation import suffixes as suf

# Core and Color Refiners 
heaPre = suf.heaIn # head of neuron
funPre = suf.funIn # (activation) function selection
posPre = suf.posIn # position argument selection


def parse_neuronNameDict_to_neuronColorDict(neuronNameDict):
    return {neuronName: [neuronNameDict[neuronName][0]] + [
        convert_candidateNames_to_colorList(candidatesList, neuronNameDict.keys()) for candidatesList in
        neuronNameDict[neuronName][1:]] for neuronName in neuronNameDict}


def convert_candidateNames_to_colorList(candidatesList, neuronNames):
    if isinstance(candidatesList, str):
        varKey, pos = candidatesList.split("=")
        return varKey + suf.disVarSuf + "=" + pos
    else:
        colorList = []
        for candidate in candidatesList:
            if candidate in neuronNames:
                colorList.append(candidate + heaPre + suf.comVarSuf)
            else:
                colorList.append(candidate + suf.disVarSuf)
        return colorList


def create_architecture(neuronNameDict, headNeuronNames=[], coreType=None):
    """
    Creates a tensor network of neuron cores with selection colors
        * neuronNameDict (in script language): Dictionary specifying to each neuronName a list of candidates (for the connective and the arguments)
        * headNeurons: List of neuronNames to be associated with hard headCores
    """
    neuronColorDict = parse_neuronNameDict_to_neuronColorDict(neuronNameDict)
    architectureCores = {}
    for neuronName in neuronColorDict.keys():
        architectureCores = {**architectureCores,
                             **create_neuron(neuronName, neuronColorDict[neuronName][0], {
                                 neuronName + posPre + str(i): posCandidates for i, posCandidates in
                                 enumerate(neuronColorDict[neuronName][1:])
                             }, coreType=coreType)}
    for headNeuronName in headNeuronNames:
        architectureCores = {**architectureCores,
                             **ch.create_boolean_head(color=headNeuronName + heaPre + suf.comVarSuf,
                                                                                             headType="truthEvaluation", coreType=coreType)}
    return architectureCores


def create_neuron(neuronName, connectiveList, candidatesDict={}, coreType=None):
    """
    Creates the cores to one neuron 
        * neuronName: String to use as prefix of the key to each core
        * connectiveList: List of connectives to be selected
        * candidatesDict: Dictionary of lists of candidates to each argument of the neuron
    """
    neuronCores = {
        neuronName + funPre + suf.selCoreIn + suf.comCoreSuf: create_connective_selectors(neuronName, candidatesDict.keys(),
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
        return {candidateKey + "_" + variables + suf.vselCoreSuf: ch.create_relational_encoding(
            inshape=[dim, dim], outshape=[2], incolors=[candidateKey + suf.selVarSuf, catName],
            outcolors=[candidateKey],
            function=selFunc, coreType=coreType,
            name=candidateKey + "_" + variables + suf.vselCoreSuf)}
    cSelectorDict = {}
    for i, variableKey in enumerate(variables):
        coreFunc = lambda c, a, o: (not (c == i)) or (a == o)
        cSelectorDict[candidateKey + "_" + variableKey + suf.vselCoreSuf] = ch.create_tensor_encoding(
            inshape=[len(variables), 2, 2], incolors=[candidateKey + suf.selVarSuf, variableKey, candidateKey],
            function=coreFunc, coreType=coreType,
            name=candidateKey + "_" + variableKey + suf.vselCoreSuf
        )
    return cSelectorDict


def create_connective_selectors(neuronName, candidateKeys, connectiveList, coreType=None):
    """
    Creates the connective selection core, using the candidateKeys as color and arity specification
    """
    if len(candidateKeys) == 1:
        return ch.create_relational_encoding(inshape=[len(connectiveList), 2], outshape=[2],
                                                 incolors=[neuronName + funPre + suf.comVarSuf, *candidateKeys],
                                                 outcolors=[neuronName + heaPre + suf.comVarSuf],
                                                 function=con.get_unary_connective_selector(connectiveList),
                                                 coreType=coreType,
                                                 name=neuronName + funPre + suf.selCoreIn + suf.comCoreSuf)
    elif len(candidateKeys) == 2:
        return ch.create_relational_encoding(inshape=[len(connectiveList), 2, 2], outshape=[2],
                                                 incolors=[neuronName + funPre + suf.comVarSuf, *candidateKeys],
                                                 outcolors=[neuronName + heaPre + suf.comVarSuf],
                                                 function=con.get_binary_connective_selector(connectiveList),
                                                 coreType=coreType,
                                                 name=neuronName + funPre + suf.selCoreIn + suf.comCoreSuf)
    else:
        raise ValueError(
            "Number {} of candidates wrong in Neuron {} with connectives {}!".format(len(candidateKeys), neuronName,
                                                                                     connectiveList))


## Auxiliary functions for application identifying the atoms and the dimension of selection variables
def find_atom_colors(specDict):
    atoms = set()
    for neuronName in specDict.keys():
        for positionList in specDict[neuronName][1:]:
            if isinstance(positionList, list):
                atoms = atoms | set([atomName + suf.disVarSuf for atomName in positionList])
    return list(atoms)


def find_selection_dimDict(specDict):
    dimDict = {}
    for neuronName in specDict:
        dimDict.update({neuronName + funPre + suf.comVarSuf: len(specDict[neuronName][0]),
                        **{neuronName + posPre + str(i) + suf.selVarSuf: len(candidates)
                           for i, candidates in enumerate(specDict[neuronName][1:])}})
    return dimDict


def find_selection_colors(specDict):
    """
    Extracts the default selection colors from a architecture dict
    """
    colors = []
    for neuronName in specDict:
        colors.append(neuronName + funPre + suf.comVarSuf)
        colors = colors + [neuronName + posPre + str(i) + suf.selVarSuf for i in
                           range(len(specDict[neuronName][1:]))]
    return colors
