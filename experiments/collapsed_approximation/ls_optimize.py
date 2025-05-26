from experiments.collapsed_approximation import slice_selection_architecture as slsel

from tnreason import engine, encoding
from tnreason.algorithms import alternating_least_squares as als

aSuf = encoding.suf.disVarSuf
varNum = 2
sparsityOrder = 2
varColList = ["a" + str(k) + aSuf for k in range(varNum)]

selCores = slsel.create_slice_selecting_tn(variableColorList=varColList, sparsityOrder=2)

parCores = {"actParCore": engine.create_basis_core(name="actParCore",
                                                   shape=[2 for _ in range(sparsityOrder)],
                                                   colors=['posNeur0_p0_vsVar', 'posNeur1_p0_vsVar'],
                                                   numberTuple=(0, 1)),
            "pCoreVar": engine.create_trivial_core(name="parCoreVar",
                                                   shape=[3 for _ in range(sparsityOrder)],
                                                   colors=['posNeur0_asVar', 'posNeur1_asVar'])
            }

tarCores = encoding.create_formulas_cores({"w1": ["imp", "a0", "a1"]})

optimizer = als.ALS(networkCores={**selCores,
                                  **parCores},
                    targetCores=tarCores,
                    importanceColors=varColList)
residuum = optimizer.alternating_optimization(["pCoreVar", "actParCore"], sweepNum=10, computeResiduum=True)

print(engine.contract(optimizer.networkCores, openColors=varColList).values)
print(engine.contract({**tarCores}, openColors=varColList).values)
