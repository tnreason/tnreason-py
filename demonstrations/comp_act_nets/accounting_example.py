from tnreason import application
from demonstrations.comp_act_nets.algorithms import moment_matching as mm
import pandas as pd

data = {
    "A1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "A2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "F": [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}
samples = pd.DataFrame(data)

formulaExpressions = {
    "(xor_A1_A2)": ["xor", "A1", "A2", 0],
    "(imp_F_A1)": ["imp", "F", "A1", 0],
}

from tnreason import engine
from tnreason.application import data_to_cores as dtc

## Calculate satisfaction rates:
satisfactionDict = {
    formulaKey + "_cV": engine.contract({**dtc.create_data_cores(samples), **application.create_cores_to_expressionsDict(
        {formulaKey: formulaExpressions[formulaKey][:-1]})}, openColors=[])[:] / samples.shape[0]
    for formulaKey in formulaExpressions
    }

matcher = mm.MomentMatcher(computationNetwork=application.create_cores_to_expressionsDict(formulaExpressions),
                           satisfactionDict=satisfactionDict, headColors=["(xor_A1_A2)_cV", "(imp_F_A1)_cV"])

matcher.alternate(iterations=1)

assert abs(matcher.canonicalParameters["(imp_F_A1)_cV"] - 1.09861228866811) < 1e-8