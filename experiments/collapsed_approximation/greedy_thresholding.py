from experiments.collapsed_approximation import slice_selection_architecture as slsel

from tnreason import engine, algorithms

import numpy as np


def iterative_greedy_steps(energyDicts, currentParameters, varColList, sparsityOrder, optimizationMethod="numpyArgMax",
                           contrastCores=[], sparsityBound=10, calibrationSweepNum=2, activePositions=[]):
    distances = np.empty(shape=(sparsityBound))
    for it in range(sparsityBound):
        _, selPosDict = greedy_gradient_step(energyDicts, currentParameters, varColList, sparsityOrder,
                                             optimizationMethod=optimizationMethod,
                                             contrastCores=contrastCores)
        activePositions = [selPosDict] + activePositions
        currentParameters, _ = update_parameters(energyDicts, currentParameters, varColList, sparsityOrder,
                                              activePositions, sweepNum=calibrationSweepNum)
        distances[it] = calculate_distance(energyDicts, currentParameters, varColList, sparsityOrder)
    return currentParameters, distances, activePositions

def get_unique_activePositions(activePostions):
    filteredActivePositions = []
    for selPosDict in activePostions:
        if selPosDict not in filteredActivePositions:
            filteredActivePositions.append(selPosDict)
    return filteredActivePositions

def value_threshold(currentParameters, activePositions, value):
    activePositions = get_unique_activePositions(activePositions)
    filteredActivePositions = []
    for selPosDict in activePositions:
        if currentParameters[selPosDict] < value:
            currentParameters[selPosDict] = - currentParameters[selPosDict]
        elif selPosDict not in filteredActivePositions:
            filteredActivePositions.append(selPosDict)
    return currentParameters, filteredActivePositions


def number_threshold(currentParameters, activePositions, remainingNumer):
    activePositions = get_unique_activePositions(activePositions)
    remainingPositions = [value[0] for _, value in
                          sorted({i: (pos, currentParameters[pos]) for i, pos in enumerate(activePositions)}.items(),
                                 key=lambda item: item[1][1], reverse=True)[:remainingNumer]]

    filteredActivePositions = []
    for selPosDict in activePositions:
        if selPosDict in remainingPositions:
            if selPosDict not in filteredActivePositions:
                filteredActivePositions.append(selPosDict)
        else:
            currentParameters[selPosDict] = - currentParameters[selPosDict]
    return currentParameters, filteredActivePositions


def greedy_gradient_step(energyDicts, currentParameters, varColList, sparsityOrder, optimizationMethod="numpyArgMax",
                         contrastCores=[]):
    """
    Update step of a
        energyDicts: Representation of tensor to be approximated
        currentParameters: current state of the approximation, to be substracted from energyDicts
    """
    currentApproximation = engine.contract(
        {"pCore": currentParameters,
         **slsel.create_slice_selecting_tn(variableColorList=varColList, sparsityOrder=sparsityOrder)},
        openColors=varColList
    )
    gradient = engine.sum_contract(
        weightedCoreDicts=energyDicts + [(-1, {"apCore": currentApproximation})] + contrastCores,
        backCores=slsel.create_slice_selecting_tn(variableColorList=varColList,
                                                  sparsityOrder=sparsityOrder),
        openColors=slsel.get_selection_colors(sparsityOrder))
    return gradient, algorithms.core_based_optimize(gradient, optimizationMethod=optimizationMethod)


def update_parameters(energyDicts, currentParameters, varColList, sparsityOrder, selPosList, sweepNum):
    distances = np.empty(shape=(sweepNum, len(selPosList)))
    for i in range(sweepNum):
        for j, selPosDict in enumerate(selPosList):
            currentParameters[selPosDict] = calculate_coordinate_update(energyDicts, currentParameters, varColList,
                                                                        sparsityOrder, selPosDict)
            distances[i,j] = calculate_distance(energyDicts, currentParameters, varColList, sparsityOrder)
    return currentParameters, distances


def calculate_distance(energyDicts, currentParameters, varColList, sparsityOrder):
    currentApproximation = engine.contract(
        coreDict={"pCore": currentParameters,
                  **slsel.create_slice_selecting_tn(variableColorList=varColList, sparsityOrder=sparsityOrder)},
        openColors=varColList
    )
    residuum = engine.sum_contract(weightedCoreDicts=energyDicts + [(-1, {"apCore": currentApproximation})],
                                   openColors=varColList)
    return np.linalg.norm(residuum.values)


def calculate_coordinate_update(energyDicts, currentParameters, varColList, sparsityOrder, selPosDict):
    currentApproximation = engine.contract(
        coreDict={"pCore": currentParameters,
                  **slsel.create_slice_selecting_tn(variableColorList=varColList, sparsityOrder=sparsityOrder)},
        openColors=varColList
    )
    residuum = engine.sum_contract(weightedCoreDicts=energyDicts + [(-1, {"apCore": currentApproximation})],
                                   openColors=varColList)
    overlap = engine.contract(coreDict={"res": residuum, **slsel.create_slice_selecting_tn(variableColorList=varColList,
                                                                                           sparsityOrder=sparsityOrder)},
                              colorEvidenceDict=selPosDict,
                              openColors=["trivial"], dimensionDict={"trivial": 1})[0]
    worldCount = engine.contract(coreDict=slsel.create_slice_selecting_tn(variableColorList=varColList,
                                                                          sparsityOrder=sparsityOrder),
                                 colorEvidenceDict=selPosDict,
                                 openColors=["trivial"], dimensionDict={"trivial": 1})[0]
    if worldCount == 0:
        print("Contradicting formula detected!")
        return 0
    return overlap / worldCount
