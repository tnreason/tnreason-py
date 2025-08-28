from tnreason import engine, representation, reasoning

import numpy as np


def generate_CANetwork(graph, colorNum, coreType, allowedColorDict={}):
    return representation.ComputationActivationNetwork(
        computationCoreDict={
            nodeColor0 + "_" + nodeColor1: get_color_constraint_core(nodeColor0, nodeColor1, colorNum, coreType)
            for nodeColor0, nodeColor1 in graph.edges},
        featureDict={
            **{nodeColor: representation.HardPartitionFeature(featureColors=[nodeColor], shape=[colorNum]) for nodeColor
               in graph.nodes},
            **{nodeColor0 + "_" + nodeColor1: representation.PassiveFeature(
                featureColors=[],
                shape=[],
                affectedComputationCores=[
                    nodeColor0 + "_" + nodeColor1])
                for nodeColor0, nodeColor1 in graph.edges}
        },
        canParamDict=allowedColorDict,
    )


def get_color_constraint_core(nodeColor0, nodeColor1, colorNum, coreType):
    return engine.create_from_slice_iterator(colors=[nodeColor0, nodeColor1], shape=[colorNum, colorNum],
                                             sliceIterator=[(1, {})] + [(-1, {nodeColor0: i, nodeColor1: i}) for i in
                                                                        range(colorNum)], coreType=coreType)


def get_edgewise_clusterDict(graph):
    return {**{nodeColor0 + "_" + nodeColor1 + "_cluster": [nodeColor0, nodeColor1, nodeColor0 + "_" + nodeColor1] for
               nodeColor0, nodeColor1 in graph.edges},
            **{nodeColor + "_cluster": [nodeColor] for nodeColor in graph.nodes}
            }


def extract_known_colors(nodeColors, canParamDict):
    knownColorDict = {}
    for nodeColor in nodeColors:
        possibilitiesCore = engine.convert(canParamDict[nodeColor], outCoreType="NumpyCore")
        if engine.contract(coreDict={"_": possibilitiesCore},
                           openColors=[])[:] == 1:
            knownColorDict[nodeColor] = np.where(possibilitiesCore.values != 0)[0][0]
    return knownColorDict


if __name__ == "__main__":
    import networkx as nx

    colorNum = 3

    G = nx.Graph()
    G.add_nodes_from(["a", "b", "c", "d"])
    G.add_edges_from([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("a", "c")])

    colorRestrictions = {
        "a": representation.create_basis_core(name="a", shape=[colorNum], colors=["a"], numberTuple=[0]),
        "b": representation.create_basis_core(name="b", shape=[colorNum], colors=["b"], numberTuple=[1])
    }

    can = generate_CANetwork(G, colorNum, allowedColorDict=colorRestrictions, coreType="NumpyCore")

    propagator = reasoning.get_inferer("ExpectationPropagator")(clusterDict=get_edgewise_clusterDict(G),
                                                                meanParamDict={
                                                                    nodeColor: engine.create_from_slice_iterator(
                                                                        shape=[colorNum], colors=[nodeColor],
                                                                        sliceIterator=[(1, {})]) for nodeColor in
                                                                    G.nodes
                                                                },
                                                                caNetwork=can)

    print("Known colors before propagation: {}".format(
        extract_known_colors(list(G.nodes), propagator.caNetwork.canParamDict)))

    propagator.propagate_until_convergence(nonTrivialFeatureKeys=list(colorRestrictions.keys()), verbose=True)

    print("Known colors after propagation: {}".format(
        extract_known_colors(list(G.nodes), propagator.caNetwork.canParamDict)))
