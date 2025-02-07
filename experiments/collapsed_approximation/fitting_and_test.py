from experiments.collapsed_approximation import slice_selection_architecture as slsel
from experiments.collapsed_approximation import greedy_thresholding as grt

from tnreason import encoding, engine

aSuf = encoding.suf.atomicVariableSuffix
varNum = 2
sparsityOrder = 2
varColList = ["a" + str(k) + aSuf for k in range(varNum)]

selColors = slsel.get_selection_colors(sparsityOrder)
selShape = [value for _, value in slsel.get_selection_dimDict(sparsityOrder, varNum).items()]

# Energy is a slice
energyDict = [(1, encoding.create_formulas_cores(expressionsDict={"w1": ["and", "a0", "a1"]}))]
currentParameters = engine.get_core("NumpyCore")(
        colors=slsel.get_selection_colors(sparsityOrder),
        shape=selShape)  # empty initialization

# Takes the trivial slice first and then pics the true slice
currentParameters, distances, activePositions = grt.iterative_greedy_steps(energyDicts=energyDict,
        currentParameters=currentParameters,
        varColList=varColList,
        sparsityOrder=sparsityOrder,
        sparsityBound=10)



print(distances)
print(currentParameters.values)
print(grt.value_threshold(currentParameters, activePositions, 0.1))
print(currentParameters.values)
