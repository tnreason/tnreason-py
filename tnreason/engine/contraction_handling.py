import numpy as np

defaultContractionMethod = "NumpyEinsum"


def contract(coreDict, openColors, dimDict={}, method=None, coreType=None):
    """
    Contractors are initialized with
        * coreDict: Dictionary of colored tensor cores specifying a network
        * openColors: List of colors to leave open in the contraction
        * dimDict: Dictionary of dimension to each color, required only when colors do not appear in the cores
        * method:
        * coreType: Required for the empty Initialization
    """
    if method is None:
        method = defaultContractionMethod

    ## Handling trivial colors (not appearing in coreDict)
    from tnreason.engine.auxiliary_cores import create_trivial_core
    dimDict.update({color: 2 for color in openColors if color not in dimDict})

    if len(coreDict) == 0:
        return create_trivial_core(name="Contracted", shape=[dimDict[color] for color in openColors], colors=openColors,
                                   coreType=coreType)

    appearingColors = list(set().union(*[coreDict[coreKey].colors for coreKey in coreDict]))
    for color in openColors:
        if color not in appearingColors:
            coreDict[color + "_trivialCore"] = create_trivial_core(color + "_trivialCore", shape=[dimDict[color]],
                                                                   colors=[color], coreType=coreType)

    ## Einstein Summation Contractors
    if method == "NumpyEinsum":
        from tnreason.engine.workload_to_numpy import NumpyEinsumContractor
        return NumpyEinsumContractor(coreDict=coreDict, openColors=openColors).contract()
    elif method == "TensorFlowEinsum":
        from tnreason.engine.workload_to_tensorflow import TensorFlowContractor
        return TensorFlowContractor(coreDict=coreDict, openColors=openColors).einsum().to_NumpyCore()
    elif method == "TorchEinsum":
        from tnreason.engine.workload_to_torch import TorchContractor
        return TorchContractor(coreDict=coreDict, openColors=openColors).einsum().to_NumpyCore()
    elif method == "TentrisEinsum":
        from tnreason.engine.workload_to_tentris import HypertrieContractor
        return HypertrieContractor(coreDict=coreDict, openColors=openColors).einsum()

    ## Variable Elimination Contractors
    elif method == "PgmpyVariableEliminator":
        from tnreason.engine.workload_to_pgmpy import PgmpyVariableEliminator
        return PgmpyVariableEliminator(coreDict=coreDict, openColors=openColors).contract()

    ## Corewise Contractor
    elif method == "CorewiseContractor":
        """
        Requires the contract_with() method of cores
        """
        return CorewiseContractor(coreDict=coreDict, openColors=openColors).contract()


    else:
        raise ValueError("Contractor Type {} not known.".format(method))


def normate(coreDict, outColors, inColors, dimDict={}, method=None, coreType=None):
    contracted = contract(coreDict, openColors=outColors + inColors, dimDict=dimDict, method=method, coreType=coreType)
    # Need to clone in order to avoid cross reference manipulation!
    sliceNorms = contract({"rawCon": contracted.clone()}, openColors=inColors, dimDict=dimDict, method=method,
                          coreType=coreType)
    for x in np.ndindex(tuple(sliceNorms.shape)):
        if sliceNorms[x] == 0:
            print("Slice {} cannot be normated!".format(x))
        else:
            contracted.multiply(1 / sliceNorms[x], {color: x[i] for i, color in enumerate(inColors)})
    return contracted


class CorewiseContractor:

    def __init__(self, coreDict={}, openColors=[]):
        self.coreDict = coreDict
        self.openColors = openColors

    def contract(self):
        ## Without optimization -> Can apply optimization from version0
        coreKeys = list(self.coreDict.keys())
        name, resultCore = coreKeys[0], self.coreDict[coreKeys[0]]
        for key in coreKeys[1:]:
            resultCore = resultCore.contract_with(self.coreDict[key])
        resultCore.reduce_colors(self.openColors)
        return resultCore
