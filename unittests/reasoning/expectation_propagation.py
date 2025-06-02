from tnreason import application, reasoning

caNetwork = application.HybridKnowledgeBase(
    weightedFormulas={"f1": ["imp", "a", "b", 2],
                      "f4": ["imp","a","b", -2],
                      "f2": ["a", 1],
                      "f3": ["b", 2]}
).create_caNetwork()

epInferer = reasoning.get_inferer("ExpectationPropagator")(caNetwork=caNetwork,
                                                           clusterDict={"c1": ["f1", "f2"], "c2": ["f1", "f3"], "c5": ["f4","f1"],
                                                                        "c3": ["f1"], "c4": ["f3"]},
                                                           )

assert len(epInferer.clusterParents["c1"]) == 0
print(epInferer.messageDict)
assert "c1" in epInferer.clusterParents["c3"]

epInferer.compute_canParam_message("c5", "c3")
print(epInferer.messageDict)
assert epInferer.messageDict["c3"]["c5"]["f1"] == 0 ## Since 2 and -2 effectively neutralize

epInferer.compute_canParam_message("c2", "c3")
print(epInferer.messageDict)
assert epInferer.messageDict["c3"]["c2"]["f1"] > 0 ## Since enforcing b communicated the canonical parameter of the implementation on b


epInferer.compute_canParam_message("c2", "c4")
print(epInferer.messageDict)
epInferer.compute_canParam_message("c1", "c3")
print(epInferer.messageDict)
epInferer.compute_canParam_message("c5", "c3")
print(epInferer.messageDict)
epInferer.compute_canParam_message("c2", "c3")
print(epInferer.messageDict)
epInferer.compute_canParam_message("c1", "c3")
print(epInferer.messageDict)
