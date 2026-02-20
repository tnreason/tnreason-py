from tnreason import representation, engine

from tnreason.representation import suffixes as suf

# Core and Color Refiners 
heaPre = suf.heaIn  # head of neuron
funPre = suf.funIn  # (activation) function selection
posPre = suf.posIn  # position argument selection


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
        architectureCores[headNeuronName + suf.actCoreSuf] = representation.create_basis_core(
            name=headNeuronName + heaPre + suf.comVarSuf,
            shape=[2], colors=[headNeuronName + heaPre + suf.comVarSuf], numberTuple=(1))

    return architectureCores


def create_neuron(neuronName, connectiveList, candidatesDict={}, coreType=None):
    """
    Creates the cores to one neuron 
        * neuronName: String to use as prefix of the key to each core
        * connectiveList: List of connectives to be selected
        * candidatesDict: Dictionary of lists of candidates to each argument of the neuron
    """
    neuronCores = {
        neuronName + funPre + suf.selCoreIn + suf.comCoreSuf: create_connective_selectors(neuronName,
                                                                                          candidatesDict.keys(),
                                                                                          connectiveList,
                                                                                          coreType=coreType)}
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
        # Case of "catVariable=[dim]", where the atomization variables of a categorical are selected
        catName, dimBracket = variables.split("=")
        dim = int(dimBracket.split("[")[1][:-1])

        return {candidateKey + "_" + variables + suf.vselCoreSuf: engine.create_from_slice_iterator(
            shape=[2, dim, dim], colors=[candidateKey, catName, candidateKey + suf.selVarSuf],
            sliceIterator=[(1, {candidateKey: 0})] + [
                (-1, {candidateKey: 0, catName: i, candidateKey + suf.selVarSuf: i}) for i in range(dim)] + [
                              (1, {candidateKey: 1, catName: i, candidateKey + suf.selVarSuf: i}) for i in range(dim)],
            coreType=coreType, name=candidateKey + "_" + variables + suf.vselCoreSuf
        )}

        # selFunc = lambda s, c: [c == s]  # Whether selection variable coincides with control variable
        # return {
        #     candidateKey + "_" + variables + suf.vselCoreSuf: tnreason.representation.basis_calculus.create_basis_encoding_from_lambda(
        #         inshape=[dim, dim], outshape=[2], incolors=[candidateKey + suf.selVarSuf, catName],
        #         outcolors=[candidateKey],
        #         indicesToIndicesFunction=selFunc, coreType=coreType,
        #         name=candidateKey + "_" + variables + suf.vselCoreSuf)}

    return {candidateKey + "_" + variableKey + suf.vselCoreSuf: engine.create_from_slice_iterator(
        shape=[2, 2, len(variables)],
        colors=[candidateKey, variableKey, candidateKey + suf.selVarSuf],
        sliceIterator=[(1, {}),
                       (-1, {candidateKey: 1, variableKey: 0, candidateKey + suf.selVarSuf: i}),
                       (-1, {candidateKey: 0, variableKey: 1, candidateKey + suf.selVarSuf: i})],
        coreType=coreType,
        name=candidateKey + "_" + variableKey + suf.vselCoreSuf
    ) for i, variableKey in enumerate(variables)}

    # cSelectorDict = {}
    # for i, variableKey in enumerate(variables):
    # coreFunc = lambda c, a, o: (not (c == i)) or (a == o)
    # cSelectorDict[
    #    candidateKey + "_" + variableKey + suf.vselCoreSuf] = tnreason.representation.coordinate_calculus.create_tensor_encoding(
    #    inshape=[len(variables), 2, 2], incolors=[candidateKey + suf.selVarSuf, variableKey, candidateKey],
    #    function=coreFunc, coreType=coreType,
    #    name=candidateKey + "_" + variableKey + suf.vselCoreSuf
    # )
    # return cSelectorDict


def create_connective_selectors(neuronName, candidateKeys, connectiveList, coreType=None):
    """
    Creates the connective selection core, using the candidateKeys as color and arity specification
    """
    return representation.get_selection_augmented_boolean_computation_core(functionNames=connectiveList,
                                                                           inColors=list(candidateKeys),
                                                                           outColor=neuronName + heaPre + suf.comVarSuf,
                                                                           selectionColor=neuronName + funPre + suf.comVarSuf,
                                                                           coreType=coreType,
                                                                           name=neuronName + funPre + suf.selCoreIn + suf.comCoreSuf)


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
