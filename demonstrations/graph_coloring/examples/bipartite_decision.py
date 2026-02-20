"""
Decide via graph coloring if a given graph is bipartite.
If the graph is connected, it suffices to color one node and do message passing with color number 2.
"""

import networkx as nx
import numpy as np


def generate_bipartite_graph(nodeSet0, nodeSet1, edgeProb=0.5, seed=None):
    """
    Generates a random bipartite graph, which is not guaranteed to be connected (phase transitions occur given the edge probability).
    """
    np.random.seed(seed)

    G = nx.Graph()
    G.add_nodes_from(nodeSet0)
    G.add_nodes_from(nodeSet1)

    for node0 in nodeSet0:
        for node1 in nodeSet1:
            if np.random.binomial(1, edgeProb) == 1:
                G.add_edge(node0, node1)
    return G


def check_result(knownColorDict, nodeSet0, nodeSet1):
    for node in nodeSet0:
        if node in knownColorDict and knownColorDict[node] != 0:
            return False
    for node in nodeSet1:
        if node in knownColorDict and knownColorDict[node] != 1:
            return False
    return True


if __name__ == "__main__":
    from demonstrations.graph_coloring import graph_constraints as rep
    from tnreason import engine, reasoning, representation

    nodeSet0Size = 50
    nodeSet1Size = 70

    G = generate_bipartite_graph(["a" + str(i) for i in range(nodeSet0Size)],
                                 ["b" + str(i) for i in range(nodeSet1Size)], edgeProb=0.15, seed=42)

    colorRestrictions = {
        "a0": representation.create_basis_core(name="a0", shape=[2], colors=["a0"], numberTuple=[0])
    }

    can = rep.generate_CANetwork(G, 2, allowedColorDict=colorRestrictions, coreType="NumpyCore")

    propagator = reasoning.get_inferer("ExpectationPropagator")(clusterDict=rep.get_edgewise_clusterDict(G),
                                                                meanParamDict={
                                                                    nodeColor: engine.create_from_slice_iterator(
                                                                        shape=[2], colors=[nodeColor],
                                                                        sliceIterator=[(1, {})]) for nodeColor in
                                                                    G.nodes
                                                                },
                                                                caNetwork=can)

    print("Known colors before propagation: {}".format(
        rep.extract_known_colors(list(G.nodes), propagator.caNetwork.canParamDict)))

    propagator.propagate_until_convergence(nonTrivialFeatureKeys=list(colorRestrictions.keys()), verbose=True)

    print("Known colors after propagation: {}".format(
        rep.extract_known_colors(list(G.nodes), propagator.caNetwork.canParamDict)))

    print("The graph has been identified to be bipartite: {}".format(
        check_result(rep.extract_known_colors(list(G.nodes), propagator.caNetwork.canParamDict),
                     ["a" + str(i) for i in range(nodeSet0Size)],
                     ["b" + str(i) for i in range(nodeSet1Size)]
                     )))
