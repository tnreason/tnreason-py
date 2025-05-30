from tnreason import application, representation, engine

## Hybrid Knowledge Base Conversion
expressionsDict = {"f1": ["imp", "a", "b"], "f2": ["imp", "a", "c"], "f3": ["a"]}
hybridKB = application.HybridKnowledgeBase(
    weightedFormulas={key: expressionsDict[key] + [0.5] for key in expressionsDict},
    facts={"fact1": ["b"]},
    categoricalConstraints={"jaskuka": ["a", "bazant"]},
    evidence={"bazant" : 0})

caNet = hybridKB.to_caNetwork()


## Markov Network Conversion
valueCore = engine.get_core("PolynomialCore")(
    values=[(1, {"a": 0}), (1.24, {"a": 1, "b": 0}), (0.36, {"a": 0, "c": 4})
            ], shape=[2, 3, 5], colors=["a", "b", "c"])

trivCore = representation.create_trivial_core(name="Fun",shape=[6,7],colors=["jezyk","krecik"],coreType="PolynomialCore")

manTrivCore = representation.create_trivial_core(name="Fun",shape=[6,7],colors=["jezyk","krecik"],coreType="PolynomialCore")
manTrivCore[3,4] = 0.134587

mN = application.MarkovNetwork(coreDict={"core": valueCore,
                                         "triv":representation.create_trivial_core(name="Fun",shape=[6,7],colors=["jezyk","krecik"],coreType="PolynomialCore"),
                                         "manTriv": manTrivCore})
mnCaNet = mN.to_caNetwork()

assert "triv_mHard" not in mnCaNet.featureDict and "triv_mSoft" not in mnCaNet.featureDict
assert "manTriv_mHard" not in mnCaNet.featureDict and "manTriv_mSoft" in mnCaNet.featureDict