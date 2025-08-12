from tnreason.application import distributions as dists

formulas = {
    "f1": ["and", "a3", "a2"],
    "f2": ["imp", "a1", "a2"],
    "f3": ["a4"]
}

canParams = {
    "f1": True,
    "f2": 1,
    "f3": False
}

hln = dists.HybridLogicNetwork(formulaDict=formulas, canParamDict=canParams)
caNet = hln.create_caNetwork()
cores = caNet.create_cores()

## Check activation cores creation

assert cores["f1_can_aC"][0] == 0
assert cores["f1_can_aC"][1] == 1

assert cores["f2_can_aC"][0] != 0
assert cores["f2_can_aC"][1] != 0

assert cores["f3_can_aC"][0] == 1
assert cores["f3_can_aC"][1] == 0

## Check local adjustments

from tnreason import reasoning

alternator = reasoning.get_inferer("BackwardAlternator")(caNetwork=caNet,
                                                         meanParamDict={"f1": True, "f2": 0.5, "f3": False})
alternator.alternating_updates()

assert alternator.caNetwork.canParamDict["f1"] == True
assert alternator.caNetwork.canParamDict["f2"] == True
assert alternator.caNetwork.canParamDict["f3"] == False
