from experiments.collapsed_approximation import slice_selection_architecture as slsel

from tnreason import engine, algorithms


def greedy_gradient_step(energyDicts, currentParameters, varColList, sparsityOrder, optimizationMethod="numpyArgMax",
                         contrastCores=[]):
    """
    Update step of a
        energyDicts: Representation of tensor to be approximated
        currentParameters: current state of the approximation, to be substracted from energyDicts
    """
    currentApproximation = engine.contract(
        currentParameters,
        slsel.create_slice_selecting_tn(variableColorList=varColList, sparsityOrder=sparsityOrder)
    )
    gradient = engine.sum_contract(
        weightedCoreDicts=energyDicts + [(-1, {"cAp": currentApproximation})] + contrastCores,
        backCores=slsel.create_slice_selecting_tn(variableColorList=varColList,
                                                  sparsityOrder=sparsityOrder),
        openColors=slsel.get_selection_colors(sparsityOrder))
    return gradient, algorithms.core_based_optimize(gradient, optimizationMethod=optimizationMethod)


def update_parameters(energyDicts, currentParameters, varColList, sparsityOrder, selPosList, sweepNum):
    for _ in range(sweepNum):
        for selPosDict in selPosList:
            currentParameters[selPosDict] = calculate_coordinate_update(energyDicts, {"pCore": currentParameters}, varColList,
                                                                        sparsityOrder, selPosDict)
            print(calculate_distance(energyDicts, {"pCore": currentParameters}, varColList, sparsityOrder))

def calculate_distance(energyDicts, currentParameters, varColList, sparsityOrder):
    currentApproximation = engine.contract(
        coreDict={**currentParameters,
                  **slsel.create_slice_selecting_tn(variableColorList=varColList, sparsityOrder=sparsityOrder)},
        openColors=varColList
    )
    residuum = engine.sum_contract(weightedCoreDicts=energyDicts + [(-1, {"cAp": currentApproximation})],
                                   openColors=varColList)
    return np.linalg.norm(residuum.values)

def calculate_coordinate_update(energyDicts, currentParameters, varColList, sparsityOrder, selPosDict):
    currentApproximation = engine.contract(
        coreDict={**currentParameters,
                  **slsel.create_slice_selecting_tn(variableColorList=varColList, sparsityOrder=sparsityOrder)},
        openColors=varColList
    )
    residuum = engine.sum_contract(weightedCoreDicts=energyDicts + [(-1, {"cAp": currentApproximation})],
                                   openColors=varColList)
    overlap = engine.contract(coreDict={"res": residuum, **slsel.create_slice_selecting_tn(variableColorList=varColList,
                                                                                           sparsityOrder=sparsityOrder)},
                              colorEvidenceDict=selPosDict,
                              openColors=["trivial"], dimensionDict={"trivial": 1})[0]
    worldCount = engine.contract(coreDict=slsel.create_slice_selecting_tn(variableColorList=varColList,
                                                                          sparsityOrder=sparsityOrder),
                                 colorEvidenceDict=selPosDict,
                                 openColors=["trivial"], dimensionDict={"trivial": 1})[0]
    return overlap / worldCount


if __name__ == "__main__":
    from tnreason import encoding
    import numpy as np

    aSuf = encoding.suf.atomicVariableSuffix
    varNum = 2
    sparsityOrder = 2
    varColList = ["a" + str(k) + aSuf for k in range(varNum)]

    selColors = slsel.get_selection_colors(sparsityOrder)
    selShape = [value for _, value in slsel.get_selection_dimDict(sparsityOrder, varNum).items()]

    energyDict = [(1, encoding.create_formulas_cores(expressionsDict={"w1": ["and", "a0", "a1"]}))]
    currentParameters = engine.get_core("NumpyCore")(  # values=np.zeros(shape=[2 for _ in range(varNum)]),
        colors=slsel.get_selection_colors(sparsityOrder),
        shape=selShape)  # empty initialization

    gradient, result = greedy_gradient_step(
        energyDicts=energyDict,
        # [(1, {"tCore": engine.get_core("NumpyCore")(colors=varColList, shape=[2 for _ in range(varNum)])})],
        currentParameters={"pCore": currentParameters},
        varColList=varColList,
        sparsityOrder=sparsityOrder,
        contrastCores=[(-1 / 2, {})])

    update_parameters(energyDict, currentParameters, varColList, sparsityOrder, [result], 10)

    print(currentParameters[result])
    # currentParameters[result] = calculate_coordinate_update(energyDict, {"pCore": currentParameters}, varColList,
    #                                                         sparsityOrder, result)
    # print(currentParameters.values)
    #
    # print(currentParameters[result])
    # exit()
    # selCores = slsel.create_slice_selecting_tn(variableColorList=varColList, sparsityOrder=2)
    # print(gradient[0, 0, 1, 0])
    # print(gradient[1, 0, 1, 1])
    # print(gradient[2, 1, 2, 1])
    #
    # print(currentParameters.shape)
    # print(currentParameters.values)
    # currentParameters[result] = 1
    # print("aft")
    # print(currentParameters.values)
    # # engine.draw_factor_graph(selCores)
