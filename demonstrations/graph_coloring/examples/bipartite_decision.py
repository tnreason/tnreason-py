"""
Decide via graph coloring if a given graph is bipartite.
If the graph is connected, it suffices to color one node and do message passing with color number 2.
"""

import networkx as nx
import numpy as np


def generate_bipartite_graph(nodeSet0, nodeSet1, edgeProb=0.5, seed=None):
    np.random.seed(seed)

    G = nx.Graph()
    G.add_nodes_from(nodeSet0)
    G.add_nodes_from(nodeSet1)

    for node0 in nodeSet0:
        for node1 in nodeSet1:
            if np.random.binomial(1, edgeProb) == 1:
                G.add_edge(node0, node1)
    return G


if __name__ == "__main__":
    from demonstrations.graph_coloring import constraint_representation as rep
    from tnreason import engine, reasoning, representation

    nodeSet0Size = 10
    nodeSet1Size = 20

    G = generate_bipartite_graph(["a" + str(i) for i in range(nodeSet0Size)],
                                 ["b" + str(i) for i in range(nodeSet1Size)], edgeProb=0.5, seed=42)

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
