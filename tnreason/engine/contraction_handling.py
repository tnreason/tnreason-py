import numpy as np

defaultContractionMethod = "NumpyEinsum"


class EngineUser:
    def __init__(self, **engineSpec):
        self.coreType = engineSpec.get("coreType", "NumpyCore")
        self.contractionMethod = engineSpec.get("contractionMethod", "NumpyEinsum")
        self.dimensionDict = engineSpec.get("dimensionDict", dict())


def sum_contract(weightedCoreDicts, backCores={}, openColors=[], dimensionDict={}, contractionMethod=None,
                 coreType=None, colorEvidenceDict={}):
    if len(weightedCoreDicts) == 0:
        return contract(backCores, openColors=openColors, dimensionDict=dimensionDict,
                        contractionMethod=contractionMethod, coreType=coreType, colorEvidenceDict=colorEvidenceDict)
    else:
        contracted = weightedCoreDicts[0][0] * contract({**weightedCoreDicts[0][1], **backCores}, openColors=openColors,
                              dimensionDict=dimensionDict,
                              contractionMethod=contractionMethod,
                              coreType=coreType,
                              colorEvidenceDict=colorEvidenceDict)
        for i in range(1, len(weightedCoreDicts)):
            contracted = contracted + (weightedCoreDicts[i][0] * contract({**weightedCoreDicts[i][1], **backCores},
                                                                          openColors=openColors,
                                                                          dimensionDict=dimensionDict,
                                                                          contractionMethod=contractionMethod,
                                                                          coreType=coreType,
                                                                          colorEvidenceDict=colorEvidenceDict))
        return contracted


def contract(coreDict, openColors, dimensionDict={}, contractionMethod=None, coreType=None, colorEvidenceDict={}):
    """
    Contractors are initialized with
        * coreDict: Dictionary of colored tensor cores specifying a network
        * openColors: List of colors to leave open in the contraction
        * dimDict: Dictionary of dimension to each color, required only when colors do not appear in the cores
        * method:
        * coreType: Required for the empty Initialization
    """
    if colorEvidenceDict:
        coreDict = {key: coreDict[key].get_slice(colorEvidenceDict) for key in coreDict}

    if contractionMethod is None:
        contractionMethod = defaultContractionMethod

    ## Handling trivial colors (not appearing in coreDict)
    from tnreason.representation import create_trivial_core
    appearingColors = list(set().union(*[coreDict[coreKey].colors for coreKey in coreDict]))
    for color in openColors:
        if color not in appearingColors:
            if color not in dimensionDict:
                dimensionDict[color] = 2
                print("Color {} handled trivially, not appearing in coreDict or dimDict.".format(color))
            coreDict[color + "_trivialCore"] = create_trivial_core(name=color + "_trivialCore",
                                                                   shape=[dimensionDict[color]],
                                                                   colors=[color], coreType=coreType)

    ## Einstein Summation Contractors
    if contractionMethod == "NumpyEinsum":
        from tnreason.engine.workload_to_numpy import NumpyEinsumContractor
        return NumpyEinsumContractor(coreDict=coreDict, openColors=openColors).contract()
    elif contractionMethod == "TensorFlowEinsum":
        from tnreason.engine.workload_to_tensorflow import TensorFlowContractor
        return TensorFlowContractor(coreDict=coreDict, openColors=openColors).einsum().to_NumpyCore()
    elif contractionMethod == "TorchEinsum":
        from tnreason.engine.workload_to_torch import TorchContractor
        return TorchContractor(coreDict=coreDict, openColors=openColors).einsum().to_NumpyCore()
    elif contractionMethod == "TentrisEinsum":
        from tnreason.engine.workload_to_tentris import HypertrieContractor
        return HypertrieContractor(coreDict=coreDict, openColors=openColors).einsum()

    ## Variable Elimination Contractors
    elif contractionMethod == "PgmpyVariableEliminator":
        from tnreason.engine.workload_to_pgmpy import PgmpyVariableEliminator
        return PgmpyVariableEliminator(coreDict=coreDict, openColors=openColors).contract()

    ## Corewise Contractor
    elif contractionMethod == "CorewiseContractor":
        """
        Requires the contract_with() method of cores
        """
        return CorewiseContractor(coreDict=coreDict, openColors=openColors).contract()


    else:
        raise ValueError("Contractor Type {} not known.".format(contractionMethod))


def normalize(coreDict, outColors, inColors, dimensionDict={}, contractionMethod=None, coreType=None):
    contracted = contract(coreDict, openColors=outColors + inColors, dimensionDict=dimensionDict,
                          contractionMethod=contractionMethod, coreType=coreType)
    sliceNorms = contract({"rawCon": contracted.clone()}, openColors=inColors, dimensionDict=dimensionDict,
                          contractionMethod=contractionMethod,
                          coreType=coreType)
    if len(inColors) == 0:
        return (1 / sliceNorms[:]) * contracted

    # Need to clone in order to avoid cross reference manipulation!
    for x in np.ndindex(tuple(sliceNorms.shape)):
        if sliceNorms[x] == 0:
            print("Slice {} cannot be normalized!".format(x))
        else:
            contracted.slice_multiply(1 / sliceNorms[x], {color: x[i] for i, color in enumerate(inColors)})
    return contracted


class CorewiseContractor:

    def __init__(self, coreDict={}, openColors=[]):
        self.coreDict = {coreKey: coreDict[coreKey].clone() for coreKey in coreDict}
        self.openColors = openColors

    def contract(self):
        ## Without optimization -> Can apply optimization from version0
        coreKeys = list(self.coreDict.keys())
        name, resultCore = coreKeys[0], self.coreDict[coreKeys[0]]
        for key in coreKeys[1:]:
            resultCore = resultCore.contract_with(self.coreDict[key])
        resultCore.reduce_colors(self.openColors)
        return resultCore
