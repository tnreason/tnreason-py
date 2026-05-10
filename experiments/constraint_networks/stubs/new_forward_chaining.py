from experiments.constraint_networks import constraint_networks as con
from tnreason import engine


class NewForwardChaining:
    def __init__(self, coresDict):
        self.cn = con.NewConstraintNetwork(coresDict)

    def propagate_node(self, nodeKey):
        if len(self.cn.cg.singleNodeEdges[nodeKey]) > 1:
            self.cn.add_subcontraction(contractKeys=self.cn.cg.singleNodeEdges[nodeKey], openColors=[nodeKey],
                                       dropKeys=self.cn.cg.singleNodeEdges[nodeKey],
                                       contractedName=nodeKey + "_core")
        entangledNodes = []
        for edgeKey in self.cn.cg.nodesDict[nodeKey].copy():
            if edgeKey not in self.cn.cg.singleNodeEdges[nodeKey]:
                self.cn.add_subcontraction(contractKeys=[edgeKey, self.cn.cg.singleNodeEdges[nodeKey][0]],
                                           openColors=self.cn.coresDict[edgeKey].colors,
                                           dropKeys=[edgeKey],
                                           contractedName=edgeKey)
                success, newEnt = self.cn.split_edge(edgeKey, colors0=[nodeKey],
                                                     colors1=[color for color in
                                                              self.cn.coresDict[
                                                                  edgeKey].colors
                                                              if
                                                              color != nodeKey],
                                                     names=[nodeKey + "_" + edgeKey, edgeKey])
                entangledNodes = entangledNodes + newEnt
        return entangledNodes

    def unit_propagation(self):
        queue = [nodeKey for nodeKey in self.cn.cg.singleNodeEdges if nodeKey not in self.cn.cg.disentangledNodes]
        while len(queue) > 0:
            nodeKey = queue.pop()
            entangledNodes = self.propagate_node(nodeKey)
            for newNode in entangledNodes:
                if newNode not in queue:
                    queue.append(newNode)
        for nodeKey in self.cn.cg.disentangledNodes:
            if len(self.cn.cg.singleNodeEdges[nodeKey]) > 1:
                print(nodeKey, self.cn.cg.singleNodeEdges[nodeKey])
                self.cn.add_subcontraction(contractKeys=self.cn.cg.singleNodeEdges[nodeKey], openColors=[nodeKey],
                                           dropKeys=self.cn.cg.singleNodeEdges[nodeKey],
                                           contractedName=nodeKey + "_core")
                print("Rem", nodeKey, self.cn.cg.singleNodeEdges[nodeKey])
