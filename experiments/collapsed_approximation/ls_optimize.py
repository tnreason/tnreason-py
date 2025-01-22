from experiments.collapsed_approximation import slice_selection_architecture as slsel

from tnreason import engine, encoding
from tnreason.algorithms import alternating_least_squares as als

aSuf = encoding.suf.atomicVariableSuffix
varNum = 2
sparsityOrder = 2
varColList = ["a" + str(k) + aSuf for k in range(varNum)]

selCores = slsel.create_slice_selecting_tn(variableColorList=varColList, sparsityOrder=2)
parCore = engine.create_trivial_core(name="parCore",
                                     shape=[3 for _ in range(sparsityOrder)] + [2 for _ in range(sparsityOrder)],
                                     colors=slsel.get_selection_colors(sparsityOrder=2))

tarCores = encoding.create_formulas_cores({"w1": ["imp", "a0", "a1"]})

optimizer = als.ALS(networkCores={**selCores,
                                  "pCore": parCore},
                    targetCores=tarCores,
                    importanceColors=varColList)
print(optimizer.alternating_optimization(["pCore"], sweepNum=1, computeResiduum=True))

## Should be the same, since selection architecture fits all in this case, but is not
print(engine.contract({**selCores, "pCore": optimizer.networkCores["pCore"]}, openColors=varColList).values)
print(engine.contract({**tarCores}, openColors=varColList).values)