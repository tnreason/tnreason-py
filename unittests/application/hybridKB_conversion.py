from tnreason import application, representation, engine

## Hybrid Knowledge Base Conversion
expressionsDict = {"f1": ["imp", "a", "b"], "f2": ["imp", "a", "c"], "f3": ["a"]}
hybridKB = application.HybridKnowledgeBase(
    weightedFormulas={key: expressionsDict[key] + [0.5] for key in expressionsDict},
    facts={"fact1": ["b"]},
    categoricalConstraints={"jaskuka": ["a", "bazant"]},
    evidence={"bazant" : 0})

caNet = hybridKB.create_caNetwork()

# Differing only when hybridKB.create_cores not using caNet (which will be the case)
caCores = caNet.create_cores()
directedCores = hybridKB.create_cores()


caComputed = engine.contract(caCores, openColors=["a", "b", "c"], contractionMethod="NumpyEinsum")
dirComputed = engine.contract(directedCores, openColors=["a", "b", "c"], contractionMethod="NumpyEinsum")
assert caComputed[0,0,0] == dirComputed[0,0,0]
assert caComputed[0,0,1] == dirComputed[0,0,1]
assert caComputed[0,1,0] == dirComputed[0,1,0]
assert caComputed[1,1,1] == dirComputed[1,1,1]


caCores = caNet.create_energyDict()
dirEnergy = hybridKB.get_energy_dict()

caComputed = engine.sum_contract(weightedCoreDicts=list(caCores.values()), openColors=["a", "b", "c"])
dirComputed = engine.sum_contract(weightedCoreDicts=list(dirEnergy.values()), openColors=["a", "b", "c"])

assert caComputed[0,0,0] == dirComputed[0,0,0]
assert caComputed[0,0,1] == dirComputed[0,0,1]
assert caComputed[0,1,0] == dirComputed[0,1,0]
assert caComputed[1,1,1] == dirComputed[1,1,1]


## Markov Network Conversion
valueCore = engine.get_core("PolynomialCore")(
    values=[(1, {"jezyk": 0}), (1.24, {"a": 1, "b": 0}), (0.36, {"a": 0, "c": 4})
            ], shape=[2, 3, 5, 6], colors=["a", "b", "c", "jezyk"])

trivCore = representation.create_trivial_core(name="Fun",shape=[6,7],colors=["jezyk","krecik"],coreType="PolynomialCore")

manTrivCore = representation.create_trivial_core(name="Fun",shape=[6,7],colors=["jezyk","krecik"],coreType="PolynomialCore")
manTrivCore[3,4] = 0.134587

coreDict={"core": valueCore,
          "triv":representation.create_trivial_core(name="Fun",shape=[6,7],colors=["jezyk","krecik"],coreType="PolynomialCore"),
          "manTriv": manTrivCore}

mN = application.MarkovNetwork(coreDict={coreKey : engine.convert(coreDict[coreKey],"NumpyCore") for coreKey in coreDict})
mnCaNet = mN.create_caNetwork()

assert "triv_mHard" not in mnCaNet.featureDict and "triv_mSoft" not in mnCaNet.featureDict
assert "manTriv_mHard" not in mnCaNet.featureDict and "manTriv_mSoft" in mnCaNet.featureDict

directeCores = mN.create_cores()
caCores = mnCaNet.create_cores()

dCore = engine.contract(directeCores, openColors=["jezyk", "krecik"], contractionMethod="NumpyEinsum")
caComputed = engine.contract(caCores, openColors=["jezyk", "krecik"], contractionMethod="NumpyEinsum")

assert dCore[2,3] == caComputed[2,3]
assert dCore[0,1] == caComputed[0,1]
