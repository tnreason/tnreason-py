from experiments.collapsed_approximation import slice_selection_architecture as slsel

from tnreason import engine, algorithms

def greedy_gradient_step(energyDicts, currentParameters, varColList, sparsityOrder=2, optimizationMethod="numpyArgMax", contrastCores=[]):
    """
    Update step of a
        energyDicts: Representation of tensor to be approximated
        currentParameters: current state of the approximation, to be substracted from energyDicts
    """
    currentApproximation = engine.contract(
        currentParameters,
        slsel.create_slice_selecting_tn(variableColorList=varColList, sparsityOrder=sparsityOrder)
    )
    gradient = engine.sum_contract(weightedCoreDicts=energyDicts + [(-1, {"cAp": currentApproximation})] + contrastCores,
                                   backCores=slsel.create_slice_selecting_tn(variableColorList=varColList,
                                                                             sparsityOrder=sparsityOrder),
                                   openColors=slsel.get_selection_colors(sparsityOrder))
    return gradient, algorithms.core_based_optimize(gradient, optimizationMethod=optimizationMethod)


if __name__ == "__main__":
    from tnreason import encoding
    import numpy as np

    aSuf = encoding.suf.atomicVariableSuffix
    varNum = 2
    sparsityOrder = 2
    varColList = ["a" + str(k) + aSuf for k in range(varNum)]

    selColors = slsel.get_selection_colors(sparsityOrder)
    selShape = [value for _, value in slsel.get_selection_dimDict(sparsityOrder, varNum).items()]

    energyDict = [(1, encoding.create_formulas_cores(expressionsDict={"w1" : ["and","a0","a1"]}))]
    currentParameters = engine.get_core("NumpyCore")(#values=np.zeros(shape=[2 for _ in range(varNum)]),
                                                     colors=slsel.get_selection_colors(sparsityOrder),
                                                     shape=selShape)

    gradient, result = greedy_gradient_step(
        energyDicts=energyDict,#[(1, {"tCore": engine.get_core("NumpyCore")(colors=varColList, shape=[2 for _ in range(varNum)])})],
        currentParameters={"pCore": currentParameters},
        varColList=varColList,
        sparsityOrder=sparsityOrder,
        contrastCores=[(-1/2,{})])

    print(result)
    selCores = slsel.create_slice_selecting_tn(variableColorList=varColList, sparsityOrder=2)
    print(gradient[0,0,1,0])
    print(gradient[1,0,1,1])
    print(gradient[2,1,2,1])

    print(currentParameters.shape)
    print(currentParameters.values)
    currentParameters[result] = 1
    print("aft")
    print(currentParameters.values)
    #engine.draw_factor_graph(selCores)
