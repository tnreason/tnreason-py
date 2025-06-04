import unittest

from tnreason import reasoning, engine, application, representation
from experiments.variable_elimination import generate_random_mn as gr
from experiments.variable_elimination import generate_clusterTree as gc

from tnreason.application.distributions import mnSoftFeatureSuffix

clusterSuf = ""

class VariableEliminationTest(unittest.TestCase):

    def test_message_schedule(self):
        hyperedgeDict = {
            "e1": ["a", "b"],
            "v1": ["a"],
            "v2": ["b"],
            "e2": ["a", "b", "c"]
        }

        clusterHyperedgeDict, clusterVariablesDict, successorDict, precessorDict = gc.generateClusterTree(hyperedgeDict,
                                                                                                          variableEliminationList=[
                                                                                                              "a", "c",
                                                                                                              "b"],
                                                                                                          clusterSuf=clusterSuf)
        mesSched = gc.find_message_schedule_towards_root(successorDict, precessorDict)

        self.assertTrue(mesSched[0] == ('a' + clusterSuf, 'c' + clusterSuf))
        self.assertTrue(mesSched[1] == ('c' + clusterSuf, 'b' + clusterSuf))

    def test_random_contraction(self):

        hyperedgeDict = {
            "e1": ["a", "b"],
            "e2": ["a"],
            "e3": ["b"],
            "e4": ["b", "c"],
            "e5": ["c"],
        }
        clusterHyperedgeDict, clusterVariablesDict, successorDict, precessorDict = gc.generateClusterTree(hyperedgeDict,
                                                                                                          variableEliminationList=[
                                                                                                              "a", "c",
                                                                                                              "b"],
                                                                                                          clusterSuf=clusterSuf)
        mesSched = gc.find_message_schedule_towards_root(successorDict, precessorDict)
        self.assertTrue(mesSched[0] == ('a' + clusterSuf, 'b' + clusterSuf))
        self.assertTrue(mesSched[1] == ('c' + clusterSuf, 'b' + clusterSuf))

        shapeDict = {"a": 2, "b": 3, "c": 5}
        ## Prepare CA Network: Including message dictionaries between clusters
        addFeatureDict = {
            varKey + "_mesTo_" + varKey2: representation.SoftPartitionFeature(
                featureColors=[variableKey for variableKey in clusterVariablesDict[varKey]
                               if variableKey in clusterVariablesDict[varKey2]],
                shape = [shapeDict[variableKey] for variableKey in clusterVariablesDict[varKey] if variableKey in clusterVariablesDict[varKey2]])
            for varKey, varKey2 in mesSched
        }
        randomTN = gr.generate_random_positive_tn(
            hyperedgeDict=hyperedgeDict,
            shapeDict=shapeDict
        )
        caNetwork = application.MarkovNetwork(randomTN).create_caNetwork()
        caNetwork.include_features(addFeatureDict)

        clusterDict = {
            **{varKey: [edgeKey + mnSoftFeatureSuffix for edgeKey in clusterHyperedgeDict[varKey]] for varKey in
               clusterHyperedgeDict},
            **{varKey + "_mesTo_" + varKey2: [varKey + "_mesTo_" + varKey2] for varKey, varKey2 in mesSched}
        }

        for varKey, varKey2 in mesSched:
            clusterDict[varKey].append(varKey + "_mesTo_" + varKey2)
            clusterDict[varKey2].append(varKey + "_mesTo_" + varKey2)

        inferer = reasoning.get_inferer("ExpectationPropagator")(
            caNetwork=caNetwork,
            clusterDict=clusterDict,
            # startMessageSchedule=mesSched
        )

        for sendCluster, arriveCluster in mesSched:
            inferer.compute_canParam_message(sendCluster, sendCluster + "_mesTo_" + arriveCluster)

        ## Check message c to b along feature c_mesTo_b being trivial with color b
        checkContraction = 1 / engine.contract({key: randomTN[key] for key in ["e4", "e5"]}, openColors=[])[:] \
                           * engine.contract({key: randomTN[key] for key in ["e4", "e5"]}, openColors=["b"])

        for i in range(shapeDict["b"]):
            self.assertTrue(abs(checkContraction.values[i] - inferer.meanParamDict["c_mesTo_b"].values[i]) < 1e-5)

        ## Check message a to b along feature a_mesTo_b being trivial with color b
        checkContraction = 1 / engine.contract({key: randomTN[key] for key in ["e1", "e2"]}, openColors=[])[:] \
                           * engine.contract({key: randomTN[key] for key in ["e1", "e2"]}, openColors=["b"])

        for i in range(shapeDict["a"]):
            self.assertTrue(abs(checkContraction.values[i] - inferer.meanParamDict["a_mesTo_b"].values[i]) < 1e-5)
