from demonstrations.graph_coloring import graph_constraints as rep
from tnreason.reasoning import message_passing as mp

from tnreason import representation, engine


def get_forward_mp_inferer(graph, colorNum, colorRestrictions):
    messageClusters = {nodeColor + "_cluster": [nodeColor] for nodeColor in graph.nodes}
    inferenceClusters = {
        nodeColor0 + "_" + nodeColor1 + "_cluster": [nodeColor0, nodeColor1, nodeColor0 + "_" + nodeColor1] for
        nodeColor0, nodeColor1 in graph.edges}
    return mp.ForwardMessagePasser(
        caNetwork=rep.generate_CANetwork(G, colorNum,
                                         allowedColorDict={k: colorRestrictions[k] for k in colorRestrictions},
                                         coreType="NumpyCore"),
        messageClusters=messageClusters,
        meanParamDict={
            nodeColor: engine.create_from_slice_iterator(
                shape=[colorNum], colors=[nodeColor],
                sliceIterator=[(1, {})]) for nodeColor in
            graph.nodes},
        inferenceClusters=inferenceClusters
    )


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

    propagator = get_forward_mp_inferer(G, colorNum, colorRestrictions)
    propagator.propagate_until_convergence(nonTrivialFeatureKeys=list(colorRestrictions.keys()), verbose=True)

    print(rep.extract_known_colors(list(G.nodes), propagator.caNetwork.canParamDict))
