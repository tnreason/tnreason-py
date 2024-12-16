from tnreason import engine

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
    from tnreason.algorithms import energy_based_algorithms as eba
    """
    Mixes two steps! Approximation (EnergyMeanFieldApproximator) and Sample Drawing (EnergyGibbsSampleCore)
    """
    if optimizationMethod == gibbsMethodString:
        sampler = eba.EnergyGibbsSampleCore(energyDict=energyDict, **specDict)
        sampler.draw_sample()
        return sampler.sample
    elif optimizationMethod == meanFieldMethodString:
        approximator = eba.NaiveMeanFieldApproximator(energyDict=energyDict, **specDict)
        approximator.anneal(
            approximationTemperatureList=specDict.get("approximationTemperatureList", [1 for _ in range(10)]))

        # return EnergyGibbsSampleCore(energyDict=approximator.get_energyDict(), **specDict)
        return approximator.draw_sample()
    elif optimizationMethod == energyMaximumMethodString:
        """
        Potential: Sparse Cores -> HUBO!
        """
        return engine.sum_contract(eba.energyDict_to_weightedCoresDicts(energyDict),
                                   openColors=specDict["variableList"],
                                   coreType=specDict.get("coreType", None),
                                   contractionMethod=specDict.get("contractionMethod", None),
                                   dimensionDict=specDict.get("dimensionDict", dict())
                                   ).get_argmax()
    elif optimizationMethod == klMaximumMethodString:
        posPhase = engine.contract(energyDict["pos"][1],
                                   openColors=specDict["variableList"]).multiply(energyDict["pos"][0])
        negPhase = engine.contract(energyDict["neg"][1],
                                   openColors=specDict["variableList"]).multiply(-energyDict["neg"][0])
        return core_based_optimize(posPhase.calculate_coordinatewise_kl_to(negPhase), "numpyArgMax")

    else:
        raise ValueError("Energy Optimization Method {} not implemented.".format(energyMaximumMethodString,
                                                                                 optimizationMethod))


def core_based_optimize(cores, optimizationMethod, **specDict):
    return core_based_optimize(engine.contract(cores, openColors=specDict["variableList"]),
                               optimizationMethod, **specDict)


def core_based_optimize(core, optimizationMethod, **specDict):
    """
        For Numpy Core + argmax, PolynomialCore + gurobi/ILP solvers
    """
    if optimizationMethod == gurobiMethodString:
        core = engine.convert(core, "PolynomialCore")
        return core.get_argmax()
    elif optimizationMethod == numpyMethodString:
        core = engine.convert(core, "NumpyCore")
        import numpy as np
        return {core.colors[i]: maxPos for i, maxPos in
                enumerate(np.unravel_index(np.argmax(core.values.flatten()), core.values.shape))}
    return core.get_argmax(optimizationMethod)
