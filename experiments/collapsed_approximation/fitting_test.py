from experiments.collapsed_approximation import slice_selection_architecture as slsel

from tnreason import engine, encoding

import numpy as np

aSuf = encoding.suf.disVarSuf
varNum = 2
sparsityOrder = 2
varColList = ["a" + str(k) + aSuf for k in range(varNum)]

tarCores = encoding.create_formulas_cores({"w1": ["imp", "a0", "a1"]})

selCores = slsel.create_slice_selecting_tn(variableColorList=varColList, sparsityOrder=2)
parCore = engine.get_core()(colors=slsel.get_selection_colors(sparsityOrder=2),
                            values=np.zeros(
                                shape=[3 for _ in range(sparsityOrder)] + [2 for _ in range(sparsityOrder)]),
                            name="pCore")

tarValues = np.random.random(size=(2, 2))

parCore[{'posNeur0_asVar': 0, 'posNeur1_asVar': 0, 'posNeur0_p0_vsVar': 0, 'posNeur1_p0_vsVar': 1}] = tarValues[0, 0]
parCore[{'posNeur0_asVar': 0, 'posNeur1_asVar': 1, 'posNeur0_p0_vsVar': 0, 'posNeur1_p0_vsVar': 1}] = tarValues[0, 1]
parCore[{'posNeur0_asVar': 1, 'posNeur1_asVar': 0, 'posNeur0_p0_vsVar': 0, 'posNeur1_p0_vsVar': 1}] = tarValues[1, 0]
parCore[{'posNeur0_asVar': 1, 'posNeur1_asVar': 1, 'posNeur0_p0_vsVar': 0, 'posNeur1_p0_vsVar': 1}] = tarValues[1, 1]

reproduced = engine.contract({**selCores, "tC": parCore}, openColors=varColList)

assert reproduced[0, 0] == tarValues[0, 0]
assert reproduced[0, 1] == tarValues[0, 1]
assert reproduced[1, 0] == tarValues[1, 0]
assert reproduced[1, 1] == tarValues[1, 1]
