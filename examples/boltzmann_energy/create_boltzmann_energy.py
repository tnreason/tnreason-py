from tnreason import representation, engine
import numpy as np

"""
Creates a tensor network decomposing the energy tensor of a Boltzmann machine, treating all variables as being open
"""


def get_boltzmann_neuronNameDict(atomList, neurKey):
    """
    Provides a names dictionary of a logical neuron parametrizing boltzmann energies
    """
    return {neurKey: [["eq", "lpas"], atomList, atomList]}


def get_boltzmann_selectionNetwork(atomList, neurKey="boltzmannNeur", coreType="NumpyCore"):
    return representation.create_architecture(get_boltzmann_neuronNameDict(atomList, neurKey), headNeuronNames=[neurKey],
                                              coreType=coreType)


def get_boltzmann_parameterCore(weightMatrix, biasVector, neurKey="boltzmannNeur", coreType="NumpyCore"):
    """
    Provides a parameter tensor representation an instance of a boltzmann machine with a weightMatrix and a potent
    """
    weightIterator = [(weightMatrix[i, j], {neurKey + "_asVar": 0, neurKey + "_p0_vsVar": i, neurKey + "_p1_vsVar": j})
                      for i, j in np.ndindex(weightMatrix.shape)]
    biasIterator = [
        (biasVector[i], {neurKey + "_asVar": 1, neurKey + "_p0_vsVar": i, neurKey + "_p1_vsVar": 0})
        for i in np.ndindex(weightMatrix.shape[0])]
    return engine.create_from_slice_iterator(shape=[2] + list(weightMatrix.shape),
                                             colors=[neurKey + "_asVar", neurKey + "_p0_vsVar", neurKey + "_p1_vsVar"],
                                             sliceIterator=weightIterator + biasIterator,
                                             coreType=coreType)


def reduce_matrix_to_upperOffDiagonalTriangle(matrix):
    ## Those are tautologies
    for i in range(atomorder):
        matrix[i, i] = 0

    ## Those are redundant by symmetry of bicond
    for i in range(atomorder):
        for j in range(i):
            matrix[i, j] = 0
    return matrix


if __name__ == "__main__":
    """
    Test instance of weights and potentials
    """
    atomorder = 2
    weightValues = reduce_matrix_to_upperOffDiagonalTriangle(np.random.rand(atomorder, atomorder))

    biasValues = np.random.rand(atomorder)
    weightMatrix = engine.get_core("NumpyCore")(values=weightValues, colors=["blub", "blub2"])
    biasVector = engine.get_core("NumpyCore")(values=biasValues, colors=["blub3"])

    """
    Create a boltzmann energy decomposition
    """
    coreType = "NumpyCore"
    contractionMethod = "NumpyEinsum"
    energyCoreDict = {
        **get_boltzmann_selectionNetwork(atomList=["a" + str(i) for i in range(atomorder)], coreType=coreType),
        "parCore": get_boltzmann_parameterCore(weightMatrix, biasVector, coreType=coreType)}

    contracted = engine.contract(coreDict=energyCoreDict, openColors=["a" + str(i) for i in range(atomorder)],
                                 contractionMethod=contractionMethod)

    ## When both atoms are false, only their bicond is true
    assert contracted[0, 0] == weightValues[0, 1]
    ## When only one is true, only one atomic formula is true
    assert contracted[0, 1] == biasValues[1]
    assert contracted[1, 0] == biasValues[0]
    ## When both atoms are true, their bicond and their atomic formulas are true
    assert contracted[1, 1] == weightValues[0, 1] + biasValues[0] + biasValues[1]
