from tnreason.representation import basis_calculus as bc
from tnreason.representation import suffixes as suf


def create_categorical_cores(categoricalsDict, coreType=None, addColorSuffixes=False):
    """
    Creates a tensor network representing the constraints of
        * categoricalsDict (in colors): Dictionary of atom color lists to each categorical variable color
    """
    if addColorSuffixes:
        categoricalsDict = {catName + suf.comVarSuf: [atomName + suf.disVarSuf for atomName in categoricalsDict[catName]] for catName in categoricalsDict}

    catCores = {}
    for catName in categoricalsDict.keys():
        catCores.update(create_constraintCoresDict(categoricalsDict[catName], catName, coreType=coreType))
    return catCores


def create_constraintCoresDict(atomColors, catColor, coreType=None):
    return {
        catColor + "_" + atomColor + suf.atoCoreSuf:
            create_single_atomization(catColor, len(atomColors), i, atomColor, coreType=coreType)[
                catColor + "_" + atomColor + suf.atoCoreSuf] for i, atomColor in enumerate(atomColors)}


def create_single_atomization(catColor, catDim, position, atomColor=None, coreType=None):
    """
    Creates the relation representation of the categorical X with its atomization to the position (int).
    If the resulting atom is not named otherwise, we call it X=position.
    """
    assert position < catDim, "Position out of range of the variable {}!".format(catColor)
    if atomColor is None:
        atomColor = catColor + "=" + str(position)
    atomizer = lambda catPos: [catPos == position]
    return {catColor + "_" + atomColor + suf.atoCoreSuf:
                bc.create_relational_encoding_from_lambda(inshape=[catDim], outshape=[2], incolors=[catColor],
                                                          outcolors=[atomColor],
                                                          indicesToIndicesFunction=atomizer, coreType=coreType,
                                                          name=catColor + "_" + atomColor + suf.atoCoreSuf
                                                          )}


def create_atomization_cores(atomizationSpecs, catDimDict, coreType=None):
    atomizationCores = {}
    for atomizationSpec in atomizationSpecs:
        catName, position = atomizationSpec.split("=")
        atomizationCores.update(
            create_single_atomization(catName, catDimDict[catName], int(position), coreType=coreType))
    return atomizationCores
