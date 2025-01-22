def draw_factor_graph(coreDict, fontsize=10, title="Factor Graph", pos=None, bipartite=False, storePath=None):

    import networkx as nx
    import matplotlib.pyplot as plt

    coreNodes = list(coreDict.keys())

    colorNodes = []
    ccEdges = []
    for coreKey in coreDict:
        for color in coreDict[coreKey].colors:
            if color not in colorNodes:
                colorNodes.append(color)
            ccEdges.append([color, coreKey])

    graph = nx.DiGraph()
    graph.add_nodes_from(coreNodes)
    graph.add_nodes_from(colorNodes)
    graph.add_edges_from(ccEdges)

    if pos is None and not bipartite:
        try:
            pos = nx.planar_layout(graph)
        except nx.NetworkXException:
            pos = nx.spring_layout(graph)
    elif bipartite:
        pos = nx.bipartite_layout(graph, coreNodes)

    nx.draw_networkx_labels(graph, pos, {atom:atom for atom in coreNodes}, font_size=fontsize,
                            bbox=dict(facecolor='blue', alpha=0.2, edgecolor='black', boxstyle='round,pad=0'))
    nx.draw_networkx_labels(graph, pos, {atom:atom for atom in colorNodes}, font_size=fontsize*0.8,
                            bbox=dict(facecolor='red', alpha=0.2, edgecolor='black', boxstyle='round'))

    nx.draw_networkx_edges(graph, pos,
                           edgelist=ccEdges,
                           arrowstyle="-",
                           width=2, alpha=1, edge_color='tab:grey')
    plt.title(title)
    if storePath is not None:
        plt.savefig(storePath)
    else:
        plt.show()