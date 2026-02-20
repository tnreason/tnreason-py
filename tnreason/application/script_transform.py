from tnreason.representation import suffixes as suf

# Core and Color Refiners
heaPre = suf.heaIn # head of neuron
funPre = suf.funIn # (activation) function selection
posPre = suf.posIn # position argument selection

"""
Handles script language inputs
"""

## Handling expressions

def drop_color_suffixes_from_assignment(colorAssignments, includeComputed=True):
    """
    For distributed and computed variables: drops the suffixes from the colors to retrieve the names
        * colorAssignments: Dictionary of colors as keys and assignments as variables
    """
    nameAssignments = dict()
    for color in colorAssignments:
        if color.endswith(suf.disVarSuf):
            nameAssignments[color[:-len(suf.disVarSuf)]] = colorAssignments[color]
        elif includeComputed:
            if color.endswith(suf.comVarSuf):
                nameAssignments[color[:-len(suf.comVarSuf)]] = colorAssignments[color]
    return nameAssignments

def add_color_suffixes(nameList):
    """
    Converts a list of names into colors by adding the suffixes
        * nameList: List of names
    """
    return [name + suf.disVarSuf for name in nameList]


def get_all_atom_colors(expressionsDict):
    """
    Identifies the leafs of the expressions in the expressionsDict as atoms,
        * expressionsDict: In script language
    Output: Colors of atoms
    """
    atoms = set()
    for key in expressionsDict:
        atoms = atoms | get_atom_colors(expressionsDict[key])
    return list(atoms)


def get_atom_colors(expression):
    if isinstance(expression, str):  ## Then an atom
        return {expression + suf.disVarSuf}
    elif len(expression) == 1:  ## Then an atomic formula
        return {expression[0] + suf.disVarSuf}
    else:  ## Then a formula with connective in first position
        atoms = set()
        for subExpression in expression[1:]:
            atoms = atoms | get_atom_colors(subExpression)
        return atoms


## Functions to identify solution expressions from architecture names when candidates are selected
def create_solution_expression(neuronNameDict, selectionDict):
    """
    Replaces the candidates of neurons by solutions and returns the identified head neurons as formulas
        * neuronNameDict (in script language): Dictionary specifying the neurons
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
        rawFormulas[neuronName] = [neuronDict[neuronName][0][selectionDict[neuronName + funPre + suf.comVarSuf]]] + [
            fix_selection(neuronDict[neuronName][i],
                          selectionDict[neuronName + posPre + str(i - 1) + suf.selVarSuf])
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
            if len(heaPre + suf.comVarSuf) > 0:  # checks, whether a neuron suffix is given
                if inNeuron.split(heaPre + suf.comVarSuf)[0] in headKeys:
                    headKeys.remove(inNeuron.split(heaPre + suf.comVarSuf)[0])
            else:
                if inNeuron in headKeys:
                    headKeys.remove(inNeuron)
    return headKeys


def replace_neuronnames(currentNeuronName, fixedNeuronDict):
    """
    Replaces the current neuron with the respective expression, after iterative replacement of depending fixed neurons
    Works both for neuronColorDicts and neuronNameDicts (split turns colors and names into names)
    """
    if len(heaPre + suf.comVarSuf) > 0:  # Then need to strip the neural variable suffix of to compare with the keys
        if currentNeuronName.split(heaPre + suf.comVarSuf)[0] not in fixedNeuronDict:
            return currentNeuronName  ## Then an atom
        currentNeuron = fixedNeuronDict[currentNeuronName.split(heaPre + suf.comVarSuf)[0]].copy()
    else:
        if currentNeuronName not in fixedNeuronDict:
            return currentNeuronName  ## Then an atom
        currentNeuron = fixedNeuronDict[currentNeuronName].copy()

    currentNeuron = [currentNeuron[0]] + [replace_neuronnames(currentNeuron[i], fixedNeuronDict) for i in
                                          range(1, len(currentNeuron))]
    return currentNeuron