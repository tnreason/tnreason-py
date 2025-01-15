from tnreason import encoding, engine
import numpy as np

"""
Creates a tensor network decomposing the energy tensor of a Boltzmann machine, treating all variables as being open
"""


def get_boltzmann_neuronNameDict(atomList, neurKey):
    """
    Provides a names dictionary of a logical neuron parametrizing boltzmann energies
    """
    return {neurKey: [["and", "lpas"], atomList, atomList]}


def get_boltzmann_selectionNetwork(atomList, neurKey="boltzmannNeur"):
    return encoding.create_architecture(get_boltzmann_neuronNameDict(atomList, neurKey))


def get_boltzmann_parameterCore(weightMatrix, biasVector, neurKey="boltzmannNeur", coreType="NumpyCore"):
    """
    Provides a parameter tensor encoding an instance of a boltzmann machine with a weightMatrix and a potent
    """
    weightIterator = [(weightMatrix[i, j], {neurKey + "_asVar": 0, neurKey + "_p0_vsVar": i, neurKey + "_p1_vsVar": j})
                      for i, j in np.ndindex(weightMatrix.shape)]
    biasIterator = [
        (biasVector[i], {neurKey + "_asVar": 0, neurKey + "_p0_vsVar": i, neurKey + "_p1_vsVar": 0})
        for i in np.ndindex(weightMatrix.shape[0])]
    return engine.create_from_slice_iterator(shape=[2] + list(weightMatrix.shape),
                                             colors=[neurKey + "_asVar", neurKey + "_p0_vsVar", neurKey + "_p1_vsVar"],
                                             sliceIterator=weightIterator + biasIterator,
                                             coreType=coreType)


if __name__ == "__main__":
    """
    Test instance of weights and potentials
    """
    atomorder = 2
    values = np.random.rand(atomorder, atomorder)
    weightMatrix = engine.get_core("NumpyCore")(values=values, colors=["blub", "blub2"])
    biasVector = engine.get_core("NumpyCore")(values=np.random.rand(atomorder), colors=["blub3"])

    """
    Create a boltzmann energy decomposition
    """
    engine.draw_factor_graph(
        {**get_boltzmann_selectionNetwork(atomList=["a" + str(i) for i in range(atomorder)]),
         "parCore": get_boltzmann_parameterCore(weightMatrix, biasVector, coreType="PandasCore")}
    )
