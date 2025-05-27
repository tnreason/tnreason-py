from tnreason import engine

from tnreason import representation

aSuf = representation.suf.disVarSuf

method = {"coreType": "HypertrieCore", "contractionMethod": "TentrisEinsum"}

cores = representation.create_formulas_cores({"f1": ["and", "a", "b"]}, coreType=method["coreType"])
contractionResult = engine.contract(coreDict=cores, openColors=["a" + aSuf],
                                    contractionMethod=method["contractionMethod"])

assert contractionResult[0] == 0
assert contractionResult[1] == 1

cores = representation.create_formulas_cores({"f1": ["and", ["not", "a"], ["or", "b", "c"]]},
                                             coreType=method["coreType"])
contractionResult = engine.contract(coreDict=cores, openColors=["a" + aSuf],
                                    contractionMethod=method["contractionMethod"])

assert contractionResult[0] == 3
assert contractionResult[1] == 0

cores = representation.create_formulas_cores({"f1": ["and", "a", "b"]}, coreType=method["coreType"])
contractionResult = engine.contract(coreDict=cores, openColors=["a" + aSuf],
                                    contractionMethod=method["contractionMethod"])

npCore = engine.convert(cores["(and_a_b)_cCore"], "NumpyCore")
htCore = engine.convert(npCore, "HypertrieCore")
npCore2 = engine.convert(htCore, "NumpyCore")

assert npCore[1,0,0] == htCore[1,0,0]