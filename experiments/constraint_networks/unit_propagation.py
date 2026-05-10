from experiments.constraint_networks import constraint_networks as con
from tnreason import engine
from tnreason.engine import get_dimDict

import numpy as np


class GenericUnitPropagation:
    def __init__(self, coresDict):
        self.cn = con.ConstraintNetwork(coresDict)

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
        self.consistent = True

    def summarize_node(self, nodeKey):
        if len(self.singleNodeEdges[nodeKey]) > 1:
            self.cn.add_subcontraction(contractKeys=self.singleNodeEdges[nodeKey], openColors=[nodeKey],
                                       dropKeys=self.singleNodeEdges[nodeKey],
                                       contractedName=nodeKey + "_core")

            for edgeKey in self.singleNodeEdges[nodeKey]:
                try:
                    self.nodesDict[nodeKey].remove(edgeKey)
                except ValueError:
                    pass
            self.singleNodeEdges[nodeKey] = [nodeKey + "_core"]
            self.nodesDict[nodeKey].append(nodeKey + "_core")

    def propagate_all_singleNodeEdges(self):
        queue = [nodeKey for nodeKey in self.singleNodeEdges if nodeKey not in self.disentangledNodes]
        while len(queue) > 0 and self.consistent:
            nodeKey = queue.pop()
            disentangled, newSingleNodeEdges = self.propagate_node(nodeKey)
            for edgeKey in newSingleNodeEdges:
                if self.cn.coresDict[edgeKey].colors[0] not in queue:
                    queue.append(self.cn.coresDict[edgeKey].colors[0])
            if disentangled:
                self.disentangledNodes.append(nodeKey)
        for nodeKey in self.singleNodeEdges:
            self.summarize_node(nodeKey)

    def propagate_node(self, nodeKey):
        """
        Contract a single node edge with all hyperedges including it
        """
        if len(self.singleNodeEdges[nodeKey]) > 1:
            self.summarize_node(nodeKey)

        disentangled = True

        newSingleNodes = []
        for edgeKey in self.nodesDict[nodeKey].copy():
            if edgeKey not in self.singleNodeEdges[nodeKey]:
                self.cn.add_subcontraction(contractKeys=[edgeKey, self.singleNodeEdges[nodeKey][0]],
                                           openColors=self.cn.coresDict[edgeKey].colors,
                                           dropKeys=[edgeKey],
                                           contractedName=edgeKey)
                splitSuccess = self.cn.split_edge(edgeKey, colors0=[nodeKey],
                                                  colors1=[color for color in
                                                           self.cn.coresDict[
                                                               edgeKey].colors
                                                           if
                                                           color != nodeKey],
                                                  names=[nodeKey + "_" + edgeKey, edgeKey])

                if splitSuccess == "Inconsistent":
                    self.consistent = False
                elif splitSuccess == True:
                    self.singleNodeEdges[nodeKey].append(nodeKey + "_" + edgeKey)
                    self.nodesDict[nodeKey].remove(edgeKey)

                    if len(self.cn.coresDict[edgeKey].colors) == 1:
                        newSingleNodes.append(edgeKey)
                        if self.cn.coresDict[edgeKey].colors[0] in self.singleNodeEdges:
                            self.singleNodeEdges[self.cn.coresDict[edgeKey].colors[0]].append(edgeKey)
                        else:
                            self.singleNodeEdges[self.cn.coresDict[edgeKey].colors[0]] = [edgeKey]
                        self.singleNodeEdges[self.cn.coresDict[edgeKey].colors[0]].append(edgeKey)
                else:
                    disentangled = False
        return disentangled, newSingleNodes


def backtrack(coreDict, depth=0, maxDepth=100, verbose=True):
    chainer = GenericUnitPropagation(coreDict)
    dimDict = get_dimDict(coreDict)
    chainer.propagate_all_singleNodeEdges()
    if verbose:
        print(f"Remaining entangled nodes: {len(dimDict) - len(chainer.disentangledNodes)}")
    ## Check whether all nodes disentangled, then success is
    if not chainer.consistent:
        if verbose:
            print("Inconsistent")
        return False, None
    elif len(chainer.disentangledNodes) == len(dimDict):
        consistent = check_local_satisfiability(coreDict)
        if verbose:
            print(f"Disentangled. Consistent: {consistent}")
        return consistent, chainer.cn.coresDict
    else:
        if depth == maxDepth:
            return False, None
        for nodeKey in np.random.permutation(list(dimDict.keys())):
            if nodeKey not in chainer.disentangledNodes:
                for nodeValue in np.random.permutation(range(dimDict[nodeKey])):
                    if verbose:
                        print("Guessing node", nodeKey, nodeValue)
                    success, resCoreDict = backtrack(
                        {**coreDict, f"guessed_{nodeKey}_{nodeValue}": engine.create_from_slice_iterator(
                            colors=[nodeKey], shape=[dimDict[nodeKey]], sliceIterator=[(1, {nodeKey: nodeValue})]
                        )}, depth=depth + 1, maxDepth=maxDepth, verbose=verbose)
                    if success:
                        return True, resCoreDict
                return False, None


def backtrack_with_restarts(coreDict, maxDepth=5, maxRestarts=100, verbose=True):
    for restart in range(maxRestarts):
        if verbose:
            print(f"##### \n Restart: {restart} \n####")
        success, resCoreDict = backtrack(coreDict, depth=0, maxDepth=maxDepth, verbose=verbose)
        if success:
            return True, resCoreDict, restart
    return False, None, maxRestarts


def check_local_satisfiability(coreDict):
    return all([engine.contract({coreKey: coreDict[coreKey]}, openColors=[])[:] != 0 for coreKey in coreDict])


def get_dimDict(coreDict):
    dimDict = {}
    for edgeKey in coreDict:
        for i, nodeKey in enumerate(coreDict[edgeKey].colors):
            assert nodeKey not in dimDict or dimDict[nodeKey] == coreDict[edgeKey].shape[i]
            dimDict[nodeKey] = coreDict[edgeKey].shape[i]
    return dimDict


if __name__ == "__main__":
    coreDict = {
        "a_bas": engine.create_from_slice_iterator(
            colors=["a"], shape=[2], sliceIterator=[(1, {"a": 0})]
        ),
        "ab_ent": engine.create_from_slice_iterator(
            colors=["a", "b"], shape=[2, 3],
            sliceIterator=[(1, {"a": 0, "b": 0}), (1, {"a": 1, "b": 1}), (1, {"a": 1, "b": 2})]
        ),
        "bc_ent": engine.create_from_slice_iterator(
            colors=["c", "b"], shape=[2, 3], sliceIterator=[(1, {"c": 1, "b": 0}), (1, {"c": 0, "b": 1})]
        )
    }

    chainer = GenericUnitPropagation(coreDict)
    chainer.propagate_all_singleNodeEdges()

    assert chainer.cn.coresDict["a_core"][0] == 1
    assert chainer.cn.coresDict["a_core"][1] == 0
    assert chainer.cn.coresDict["b_core"][0] == 1
    assert chainer.cn.coresDict["b_core"][1] == 0
    assert chainer.cn.coresDict["c_core"][0] == 0
    assert chainer.cn.coresDict["c_core"][1] == 1
