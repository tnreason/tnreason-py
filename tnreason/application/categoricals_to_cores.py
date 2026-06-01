from tnreason.representation import basis_calculus as bc
from tnreason.representation import suffixes as suf

from tnreason import engine

def create_at_most_one_constraints(constraintDict, coreType=None):
    coreDict = dict()
    for decVariable in constraintDict:
        coreDict.update(create_at_most_one_constraint(constraintDict[decVariable], decVariable, coreType=coreType))
    return coreDict

def create_at_most_one_constraint(atomVariables, decVariable, coreType=None):
    return {f"{atomVariable}_{decVariable}":
                engine.create_from_slice_iterator(shape=[2, len(atomVariables) + 1],
                                                  colors=[atomVariable, decVariable],
                                                  sliceIterator=[(1, {atomVariable: 0}),
                                                                 (-1, {atomVariable: 0, decVariable: i}),
                                                                 (1, {atomVariable: 1, decVariable: i})
                                                                 ])
            for i, atomVariable in enumerate(atomVariables)}


## To be called "create_exactly_one_constraint"
def create_categorical_cores(categoricalsDict, coreType=None, addColorSuffixes=False):
    """
    Creates a tensor network representing the constraints of
        * categoricalsDict (in colors): Dictionary of atom color lists to each categorical variable color
    """
    if addColorSuffixes:
        categoricalsDict = {
            catName + suf.comVarSuf: [atomName + suf.disVarSuf for atomName in categoricalsDict[catName]] for catName in
            categoricalsDict}

    return {k: v for catName in categoricalsDict for k, v in
            create_constraintCoresDict(categoricalsDict[catName], catName, coreType=coreType).items()}


def create_constraintCoresDict(atomColors, catColor, coreType=None):
    return {catColor + "_" + atomColor + suf.atoCoreSuf:
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
    return {catColor + "_" + atomColor + suf.atoCoreSuf: engine.create_from_slice_iterator(
        shape=[2, catDim], colors=[atomColor, catColor],
        sliceIterator=[(1, {atomColor: 0}),
                       (-1, {atomColor: 0, catColor: position}),
                       (1, {atomColor: 1, catColor: position})],
        coreType=coreType, name=catColor + "_" + atomColor + suf.atoCoreSuf
    )}


def create_atomization_cores(atomizationSpecs, catDimDict, coreType=None):
    atomizationCores = {}
    for atomizationSpec in atomizationSpecs:
        catName, position = atomizationSpec.split("=")
        atomizationCores.update(
            create_single_atomization(catName, catDimDict[catName], int(position), coreType=coreType))
    return atomizationCores
