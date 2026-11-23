from experiments.constraint_networks import constraint_networks as con
from tnreason import engine
from tnreason.engine import get_dimDict

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np


@dataclass
class EdgePropagationResult:
    edgeKey: str
    splitSuccess: object
    edgeCore: object
    nodeEdgeKey: str = None
    nodeEdgeCore: object = None
    newSingleNode: str = None


class GenericForwardChaining:
    def __init__(
        self,
        coresDict,
        validate=False,
        parallel=True,
        max_workers=3,
        min_parallel_edges=2,
    ):
        """
        ``parallel=False`` keeps edge propagation serial for easier debugging
        and for workloads where thread-pool overhead dominates.
        """
        self.cn = con.ConstraintNetwork(coresDict)
        self.consistent = True
        self.validate = validate
        self.parallel = parallel
        self.max_workers = max_workers
        self.min_parallel_edges = min_parallel_edges
        self._executor = None
        self.rebuild_indices()

    def rebuild_indices(self):
        """
        Rebuild cached graph indexes from the current cores.

        Forward chaining mutates ``self.cn.coresDict`` by contracting and
        splitting edges. Keeping the node/edge maps as derived data avoids stale
        references to deleted cores and duplicate unary edge registrations.
        """
        self.nodesDict = {}
        self.singleNodeEdges = {}
        for edgeKey, core in self.cn.coresDict.items():
            for nodeKey in core.colors:
                self.nodesDict.setdefault(nodeKey, []).append(edgeKey)
            if len(core.colors) == 1:
                self.singleNodeEdges.setdefault(core.colors[0], []).append(edgeKey)

        self.disentangledNodes = [
            nodeKey
            for nodeKey, singleEdges in self.singleNodeEdges.items()
            if len(self.nodesDict.get(nodeKey, [])) == len(singleEdges)
        ]

    def validate_indices(self):
        for nodeKey, edgeKeys in self.nodesDict.items():
            for edgeKey in edgeKeys:
                assert edgeKey in self.cn.coresDict
                assert nodeKey in self.cn.coresDict[edgeKey].colors
        for nodeKey, edgeKeys in self.singleNodeEdges.items():
            for edgeKey in edgeKeys:
                assert edgeKey in self.cn.coresDict
                assert self.cn.coresDict[edgeKey].colors == [nodeKey]

    def support_core(self, core):
        values = getattr(core, "values", None)
        if values is None:
            return con.support_of(core)
        try:
            support_values = (np.asarray(values) != 0).astype(float)
            return engine.get_core(getattr(core, "coreType", None))(
                values=support_values,
                colors=list(core.colors),
                name=getattr(core, "name", "Support"),
            )
        except Exception:
            return con.support_of(core)

    def summarize_node(self, nodeKey):
        singleEdges = list(self.singleNodeEdges.get(nodeKey, []))
        if len(singleEdges) > 1:
            self.cn.add_subcontraction(contractKeys=singleEdges, openColors=[nodeKey],
                                       dropKeys=singleEdges,
                                       contractedName=nodeKey + "_core")
            self.cn.coresDict[nodeKey + "_core"] = self.support_core(
                self.cn.coresDict[nodeKey + "_core"]
            )
            self.rebuild_indices()
            if self.validate:
                self.validate_indices()

    def propagate_all_singleNodeEdges(self):
        queue = [nodeKey for nodeKey in self.singleNodeEdges if nodeKey not in self.disentangledNodes]
        try:
            while len(queue) > 0 and self.consistent:
                nodeKey = queue.pop()
                if nodeKey not in self.singleNodeEdges:
                    continue
                disentangled, newSingleNodeEdges = self.propagate_node(nodeKey)
                for newNodeKey in newSingleNodeEdges:
                    if newNodeKey not in queue and newNodeKey not in self.disentangledNodes:
                        queue.append(newNodeKey)
            for nodeKey in list(self.singleNodeEdges):
                self.summarize_node(nodeKey)
        finally:
            if self._executor is not None:
                self._executor.shutdown()
                self._executor = None

    def propagate_node(self, nodeKey):
        """
        Contract a single node edge with all hyperedges including it
        """
        if nodeKey not in self.singleNodeEdges:
            return True, []
        if len(self.singleNodeEdges[nodeKey]) > 1:
            self.summarize_node(nodeKey)
        if nodeKey not in self.singleNodeEdges:
            return True, []

        disentangled = True

        newSingleNodes = set()
        singleEdgeKey = self.singleNodeEdges[nodeKey][0]
        singleNodeEdges = list(self.singleNodeEdges[nodeKey])
        edgeKeys = []
        for edgeKey in list(self.nodesDict.get(nodeKey, [])):
            if edgeKey not in self.cn.coresDict:
                continue
            if nodeKey not in self.cn.coresDict[edgeKey].colors:
                continue
            if edgeKey not in singleNodeEdges:
                edgeKeys.append(edgeKey)

        singleEdgeCore = self.cn.coresDict[singleEdgeKey]
        edgeCores = {
            edgeKey: self.cn.coresDict[edgeKey]
            for edgeKey in edgeKeys
        }
        results = self.propagate_edge_snapshots(
            nodeKey=nodeKey,
            singleEdgeKey=singleEdgeKey,
            singleEdgeCore=singleEdgeCore,
            edgeCores=edgeCores,
        )

        for result in results:
            if result.splitSuccess == "Inconsistent":
                self.cn.coresDict[result.edgeKey] = result.edgeCore
                self.consistent = False
            elif result.splitSuccess == True:
                if result.edgeKey in self.cn.coresDict:
                    del self.cn.coresDict[result.edgeKey]
                self.cn.coresDict[result.nodeEdgeKey] = result.nodeEdgeCore
                self.cn.coresDict[result.edgeKey] = result.edgeCore
                if result.newSingleNode is not None:
                    newSingleNodes.add(result.newSingleNode)
            else:
                self.cn.coresDict[result.edgeKey] = result.edgeCore
                disentangled = False
        self.rebuild_indices()
        if self.validate:
            self.validate_indices()
        return nodeKey in self.disentangledNodes and disentangled, list(newSingleNodes)

    def propagate_edge_snapshots(self, nodeKey, singleEdgeKey, singleEdgeCore, edgeCores):
        worker_args = [(nodeKey, edgeKey, edgeCore, singleEdgeKey, singleEdgeCore)
                        for edgeKey, edgeCore in edgeCores.items()]
        if not self.parallel or len(worker_args) < self.min_parallel_edges:
            return [self.propagate_edge_snapshot(*worker_arg) for worker_arg in worker_args]
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return list(self._executor.map(self.propagate_edge_snapshot_from_args, worker_args))

    def propagate_edge_snapshot_from_args(self, worker_arg):
        return self.propagate_edge_snapshot(*worker_arg)

    def propagate_edge_snapshot(self, nodeKey, edgeKey, edgeCore, singleEdgeKey, singleEdgeCore):
        restrictedCore = engine.contract(
            {edgeKey: edgeCore, singleEdgeKey: singleEdgeCore},
            openColors=list(edgeCore.colors),
        )
        restrictedCore = self.support_core(restrictedCore)
        return self.split_edge_snapshot(
            nodeKey=nodeKey,
            edgeKey=edgeKey,
            edgeCore=restrictedCore,
        )

    def split_edge_snapshot(self, nodeKey, edgeKey, edgeCore):
        colors1 = [
            color
            for color in edgeCore.colors
            if color != nodeKey
        ]
        coordinateSum = engine.contract({edgeKey: edgeCore}, openColors=[])[:]
        if coordinateSum == 0:
            return EdgePropagationResult(
                edgeKey=edgeKey,
                splitSuccess="Inconsistent",
                edgeCore=edgeCore,
            )

        core0 = engine.contract({edgeKey: edgeCore}, openColors=[nodeKey])
        core1 = engine.contract({edgeKey: edgeCore}, openColors=colors1)
        testCore = 1 / coordinateSum * engine.contract(
            {"core0": core0, "core1": core1},
            openColors=edgeCore.colors,
        )
        if engine.cores_close(testCore, edgeCore):
            return EdgePropagationResult(
                edgeKey=edgeKey,
                splitSuccess=True,
                edgeCore=core1,
                nodeEdgeKey=nodeKey + "_" + edgeKey,
                nodeEdgeCore=1 / coordinateSum * core0,
                newSingleNode=core1.colors[0] if len(core1.colors) == 1 else None,
            )
        return EdgePropagationResult(
            edgeKey=edgeKey,
            splitSuccess=False,
            edgeCore=edgeCore,
        )


def backtrack(
    coreDict,
    depth=0,
    maxDepth=100,
    verbose=True,
    parallel=True,
    max_workers=None,
    min_parallel_edges=2,
):
    chainer = GenericForwardChaining(
        coreDict,
        parallel=parallel,
        max_workers=max_workers,
        min_parallel_edges=min_parallel_edges,
    )
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
                        )}, depth=depth + 1, maxDepth=maxDepth, verbose=verbose, parallel=parallel,
                        max_workers=max_workers, min_parallel_edges=min_parallel_edges)
                    if success:
                        return True, resCoreDict
                return False, None


def backtrack_with_restarts(
    coreDict,
    maxDepth=5,
    maxRestarts=100,
    verbose=True,
    parallel=True,
    max_workers=None,
    min_parallel_edges=2,
):
    for restart in range(maxRestarts):
        if verbose:
            print(f"##### \n Restart: {restart} \n####")
        success, resCoreDict = backtrack(
            coreDict,
            depth=0,
            maxDepth=maxDepth,
            verbose=verbose,
            parallel=parallel,
            max_workers=max_workers,
            min_parallel_edges=min_parallel_edges,
        )
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

    chainer = GenericForwardChaining(coreDict)
    chainer.propagate_all_singleNodeEdges()

    assert chainer.cn.coresDict["a_core"][0] == 1
    assert chainer.cn.coresDict["a_core"][1] == 0
    assert chainer.cn.coresDict["b_core"][0] == 1
    assert chainer.cn.coresDict["b_core"][1] == 0
    cEdge = chainer.singleNodeEdges["c"][0]
    assert chainer.cn.coresDict[cEdge][0] == 0
    assert chainer.cn.coresDict[cEdge][1] == 1
