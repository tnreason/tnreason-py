from tnreason import engine

class NewConstraintNetwork:
    """
    Informed about the node-edge adjacency, by a constraintGraph instance
    """

    def __init__(self, coresDict):
        self.coresDict = coresDict
        self.cg = ConstraintGraph(coresDict=coresDict)

    def add_subcontraction(self, contractKeys, openColors, dropKeys, contractedName):
        ## Check dropKeys: They have to be in contractKeys and the corresponding colors left open
        for dropKey in dropKeys:
            assert dropKey in contractKeys
            assert all([color in openColors for color in self.coresDict[dropKey].colors])

        # Currently not doing the support core, numbers larger than 1 in coordinates!
        self.coresDict[contractedName] = engine.contract(
            {contractKey: self.coresDict[contractKey] for contractKey in contractKeys}, openColors=openColors)

        for dropKey in dropKeys:
            self.cg.drop_edge(dropKey, self.coresDict[dropKey].colors)
        self.cg.add_edge(contractedName, openColors)

        # Delete tensors
        for key in dropKeys:
            if key != contractedName and key in self.coresDict:
                del self.coresDict[key]

    def split_edge(self, splitKey, colors0, colors1, names=None):
        """
        Returns whether the split was successful, and what colors
        """
        ## Check whether the edge can be split and split the edge if possible

        core0 = engine.contract({splitKey: self.coresDict[splitKey]}, openColors=colors0)
        core1 = engine.contract({splitKey: self.coresDict[splitKey]}, openColors=colors1)
        coordinateSum = engine.contract({splitKey: self.coresDict[splitKey]}, openColors=[])[:]

        testCore = 1 / coordinateSum * engine.contract({"core0": core0, "core1": core1},
                                                       openColors=self.coresDict[splitKey].colors)

        if names is None:
            names = [splitKey + "_0", splitKey + "_1"]

        if testCore == self.coresDict[splitKey]:
            newUnitEdges = []
            if len(colors0) == 1 and colors0[0] not in self.cg.singleNodeEdges:
                newUnitEdges.append(colors0[0])
            if len(colors1) == 1 and colors1[0] not in self.cg.singleNodeEdges:
                newUnitEdges.append(colors1[0])

            self.cg.drop_edge(splitKey, self.coresDict[splitKey].colors)
            del self.coresDict[splitKey]

            self.coresDict[names[0]] = 1 / coordinateSum * core0
            self.coresDict[names[1]] = core1
            self.cg.add_edge(names[0], colors0)
            self.cg.add_edge(names[1], colors1)

            return True, newUnitEdges
        else:
            return False, []

    def replace_by_support(self, coreKey):
        """
        Replaces a core by its support, and erase if it is trivial
        """
        if not has_trivial_support(self.coresDict[coreKey]):
            self.coresDict[coreKey] = support_of(self.coresDict[coreKey])
        else:
            del self.coresDict[coreKey]




class ConstraintGraph:
    def __init__(self, coresDict):
        ## Initialize nodesDict storing the node edge inclusions
        self.nodesDict = dict()  # Storing those edges containing the node
        self.singleNodeEdges = dict()  # Storing those edges which only contain the node
        for edgeKey in coresDict:
            for nodeKey in coresDict[edgeKey].colors:
                if nodeKey not in self.nodesDict:
                    self.nodesDict[nodeKey] = [edgeKey]
                else:
                    self.nodesDict[nodeKey].append(edgeKey)
            if len(coresDict[edgeKey].colors) == 1:
                if coresDict[edgeKey].colors[0] not in self.singleNodeEdges:
                    self.singleNodeEdges[coresDict[edgeKey].colors[0]] = [edgeKey]
                else:
                    self.singleNodeEdges[coresDict[edgeKey].colors[0]].append(edgeKey)

        self.disentangledNodes = [nodeKey for nodeKey in self.singleNodeEdges if
                                  len(self.nodesDict[nodeKey]) == len(self.singleNodeEdges[nodeKey])]

    def add_edge(self, edgeKey, colors):
        if len(colors) == 1:
            if colors[0] not in self.singleNodeEdges:
                self.singleNodeEdges[colors[0]] = [edgeKey]
            else:
                self.singleNodeEdges[colors[0]].append(edgeKey)
        for color in colors:
            if color not in self.nodesDict:
                self.nodesDict[color] = [edgeKey]
            else:
                self.nodesDict[color].append(edgeKey)
        if len(colors) > 1:
            for color in colors:
                if color in self.disentangledNodes:
                    self.disentangledNodes.remove(color)

    def drop_edge(self, edgeKey, colors):
        if len(colors) == 1:
            self.singleNodeEdges[colors[0]].remove(edgeKey)
        for color in colors:
            self.nodesDict[color].remove(edgeKey)
            if color in self.singleNodeEdges:
                if len(self.nodesDict[color]) == len(
                        self.singleNodeEdges[color]) and color not in self.disentangledNodes:
                    self.disentangledNodes.append(color)