from tnreason import engine

from tnreason.encoding import suffixes as suf


def create_categorical_cores(categoricalsDict, coreType=None):
    """
    Creates a tensor network representing the constraints of
        * categoricalsDict: Dictionary of atom lists to each categorical variable
    """
    catCores = {}
    for catName in categoricalsDict.keys():
        catCores = {**catCores, **create_constraintCoresDict(categoricalsDict[catName], catName, coreType=coreType)}
    return catCores


def create_constraintCoresDict(atomColors, catColor, coreType=None):
    return {
        catColor + "_" + atomName + suf.atomizationCoreSuffix:
            create_single_atomization(catColor, len(atomColors), i, atomName, coreType=coreType)[
                catColor + "_" + atomName + suf.atomizationCoreSuffix] for i, atomName in enumerate(atomColors)}


def create_single_atomization(catColor, catDim, position, atomColor=None, coreType=None):
    """
    Creates the relation encoding of the categorical X with its atomization to the position (int).
    If the resulting atom is not named otherwise, we call it X=position.
    """
    assert position < catDim, "Position out of range of the variable {}!".format(catColor)
    if atomColor is None:
        atomColor = catColor + "=" + str(position)
    atomizer = lambda catPos: [catPos == position]
    return {catColor + "_" + atomColor + suf.atomizationCoreSuffix:
                engine.create_relational_encoding(inshape=[catDim], outshape=[2], incolors=[catColor],
                                                  outcolors=[atomColor],
                                                  function=atomizer, coreType=coreType,
                                                  name=catColor + "_" + atomColor + suf.atomizationCoreSuffix
                                                  )}


def create_atomization_cores(atomizationSpecs, catDimDict, coreType=None):
    atomizationCores = {}
    for atomizationSpec in atomizationSpecs:
        catName, position = atomizationSpec.split("=")
        atomizationCores.update(
            create_single_atomization(catName, catDimDict[catName], int(position), coreType=coreType))
    return atomizationCores
