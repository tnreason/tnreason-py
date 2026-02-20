from tnreason import application, reasoning, engine, representation

firstPolCore = engine.get_core("PolynomialCore")(
    values=[(1, {"a": 0}), (1.24, {"a": 1, "b": 0}), (0.36, {"a": 0, "c": 4})
            ], shape=[2, 3, 5], colors=["a", "b", "c"])
firstPolCore = firstPolCore + engine.get_core("PolynomialCore")(values=[(1, {})], shape=[2, 3, 5],
                                                                colors=["a", "b", "c"])
secPolCore = engine.get_core("PolynomialCore")(
    values=[(1, {"d": 0}), (1.24, {"d": 1, "c": 0}), (0.36, {"b": 0, "c": 4})
            ], shape=[2, 3, 5], colors=["d", "b", "c"])
secPolCore = secPolCore + engine.get_core("PolynomialCore")(values=[(1, {})], shape=[2, 3, 5], colors=["d", "b", "c"])

tensorNetwork = {"core1": engine.convert(firstPolCore, "NumpyCore"),
                 "core2": engine.convert(secPolCore, "NumpyCore")}

testMN = application.MarkovNetwork(coreDict=tensorNetwork)

testCANetwork = testMN.create_caNetwork()
testCANetwork.include_features(
    featureDict={"mesFeature": representation.SoftPartitionFeature(name="mesFeature", featureColors=["b", "c"],
                                                              interpretationDict={"b": range(3), "c": range(5)})}
)

epInferer = reasoning.get_inferer("ExpectationPropagator")(caNetwork=testCANetwork,
                                                           clusterDict={"c1": ["core1_mSoft", "mesFeature"],
                                                                        "mes": ["mesFeature"],
                                                                        "c2": ["core2_mSoft", "mesFeature"]},
                                                           )
epInferer.compute_canParam_message("c1", "mes")
epInferer.compute_canParam_message("c2", "mes")

checkContract = 1 / engine.contract(coreDict=tensorNetwork, openColors=[])[:] * engine.contract(coreDict=tensorNetwork,
                                                                                                openColors=["b", "c"])

import numpy as np

for tup in np.ndindex(3,5):
    assert abs(checkContract.values[tup]-epInferer.meanParamDict["mesFeature"].values[tup])<1e-5

