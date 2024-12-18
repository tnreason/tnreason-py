from tnreason import engine
from tnreason.algorithms import energy_based_algorithms as eba

## Energy-based
gibbsMethodString = "gibbsSample"
meanFieldMethodString = "meanFieldSample"
energyMaximumMethodString = "exactEnergyMax"
klMaximumMethodString = "exactKLMax"
energyOptimizationMethods = [gibbsMethodString, meanFieldMethodString, energyMaximumMethodString, klMaximumMethodString]

## Distribution-based
gurobiMethodString = "gurobi"
numpyMethodString = "numpyArgMax"
coreOptimizationMethods = [gurobiMethodString, numpyMethodString]


def energy_based_optimize(energyDict=[], optimizationMethod=gibbsMethodString, **specDict):
    """
    specDict contains engine spec and additional arguments for the optimization
    Mixes two steps! Approximation (EnergyMeanFieldApproximator) and Sample Drawing (EnergyGibbsSampleCore)
    """
    if optimizationMethod == gibbsMethodString:
        sampler = eba.EnergyGibbsSampleCore(energyDict=energyDict, **specDict)
        sampler.draw_sample()
        return sampler.sample
    elif optimizationMethod == meanFieldMethodString:
        approximator = eba.NaiveMeanFieldApproximator(energyDict=energyDict, colors=specDict["variableList"],
                                                      **specDict)
        approximator.anneal(
            approximationTemperatureList=specDict.get("approximationTemperatureList", [1 + i for i in range(10)]))
        return approximator.get_maxima()
    elif optimizationMethod == energyMaximumMethodString:
        contracted = engine.sum_contract(eba.energyDict_to_weightedCoresDicts(energyDict),
                                   openColors=specDict["variableList"],
                                   coreType=specDict.get("coreType", None),
                                   contractionMethod=specDict.get("contractionMethod", None),
                                   dimensionDict=specDict.get("dimensionDict", dict())
                                   )
        if specDict.get("coreType",None) == "PolynomialCore":
            return core_based_optimize(contracted, "gurobi", coreConversion=False)
        elif specDict.get("coreType",None) == "NumpyCore":
            return core_based_optimize(contracted, "numpyArgMax", coreConversion=False)
        else:
            return core_based_optimize(contracted, "numpyArgMax", coreConversion=True)

    elif optimizationMethod == klMaximumMethodString:
        posDist = engine.contract(energyDict["pos"][1],
                                   openColors=specDict["variableList"]).multiply(energyDict["pos"][0])
        negDist = engine.contract(energyDict["neg"][1],
                                   openColors=specDict["variableList"]).multiply(-energyDict["neg"][0])
        return core_based_optimize(posDist.calculate_coordinatewise_kl_to(negDist), "numpyArgMax")

    else:
        raise ValueError("Energy Optimization Method {} not implemented.".format(energyMaximumMethodString,
                                                                                 optimizationMethod))

def core_based_optimize(core, optimizationMethod, coreConversion=True, **specDict):
    """
        For Numpy Core + argmax, PolynomialCore + gurobi/ILP solvers
    """
    if optimizationMethod == gurobiMethodString:
        if coreConversion:
            core = engine.convert(core, "PolynomialCore")
        return core.get_argmax()
    elif optimizationMethod == numpyMethodString:
        if coreConversion:
            core = engine.convert(core, "NumpyCore")
        import numpy as np
        return {core.colors[i]: maxPos for i, maxPos in
                enumerate(np.unravel_index(np.argmax(core.values.flatten()), core.values.shape))}
    return core.get_argmax(optimizationMethod)
