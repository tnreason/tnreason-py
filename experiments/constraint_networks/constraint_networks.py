from collections import defaultdict
import numpy as np
from tnreason import engine


class ConstraintNetwork:
    def __init__(self, coresDict, maintain_indices=False):
        self.coresDict = coresDict
        self.maintain_indices = maintain_indices
        if self.maintain_indices:
            self.rebuild_indices()

    def rebuild_indices(self):
        """Rebuild structural indices derived from ``coresDict``.

        The tensor cores remain the authoritative representation.  These
        indices only make repeated propagation and search cheap enough to
        avoid rediscovering the constraint graph at every search node.

        ``coresDict`` is public for historical reasons, so callers that
        mutate it directly must call this method afterwards.
        """
        dimension_dict = {}
        core_keys_by_color = defaultdict(list)
        colors_by_core_key = {}
        for core_key, core in self.coresDict.items():
            colors = tuple(core.colors)
            shape = tuple(core.shape)
            if len(colors) != len(shape):
                raise ValueError(
                    f"Core {core_key} has {len(colors)} colors but shape {shape}."
                )
            if len(colors) != len(set(colors)):
                raise ValueError(f"Core {core_key} contains duplicate colors: {colors}.")
            colors_by_core_key[core_key] = colors
            for color, dimension in zip(colors, shape):
                if color in dimension_dict and dimension_dict[color] != dimension:
                    raise ValueError(
                        f"Color {color} has inconsistent dimensions "
                        f"{dimension_dict[color]} and {dimension}."
                    )
                dimension_dict[color] = dimension
                core_keys_by_color[color].append(core_key)

        self.dimension_dict = dimension_dict
        self.core_keys_by_color = {color: tuple(core_keys) for color, core_keys in core_keys_by_color.items()}
        self.colors_by_core_key = colors_by_core_key
        return self

    def create_mac_state(self, domains=None):
        """Create independent reversible MAC state for this network.

        The constraint cores are never modified by the returned state, and its
        core snapshot is independent of later core-value changes.  Multiple
        states can therefore safely share one ``ConstraintNetwork``.  Call
        ``rebuild_indices`` after changing the network's keys, colors, or
        shapes; mutating only tensor values does not require a rebuild.
        """
        from experiments.constraint_networks.mac import ConstraintNetworkState

        if not self.maintain_indices:
            self.rebuild_indices()
        return ConstraintNetworkState(self, domains=domains)

    def solve_mac(self, domains=None, max_nodes=100000, *, state=None):
        """Solve with maintaining GAC search, optionally reusing a prepared state."""
        from experiments.constraint_networks.mac import ConstraintNetworkSolver

        return ConstraintNetworkSolver(self, max_nodes=max_nodes).solve(domains=domains, state=state)

    def add_subcontraction(self, contractKeys, openColors, dropKeys, contractedName):
        ## Check dropKeys: They have to be in contractKeys and the corresponding colors left open
        for dropKey in dropKeys:
            assert dropKey in contractKeys
            assert all([color in openColors for color in self.coresDict[dropKey].colors])

        # Currently not doing the support core, numbers larger than 1 in coordinates!
        self.coresDict[contractedName] = engine.contract(
            {contractKey: self.coresDict[contractKey] for contractKey in contractKeys}, openColors=openColors)
        for key in dropKeys:
            if key != contractedName and key in self.coresDict:
                del self.coresDict[key]
        if self.maintain_indices:
            self.rebuild_indices()

    def split_edge(self, splitKey, colors0, colors1, names=None):
        ## Check whether the edge can be split and split the edge if possible
        core0 = engine.contract({splitKey: self.coresDict[splitKey]}, openColors=colors0)
        core1 = engine.contract({splitKey: self.coresDict[splitKey]}, openColors=colors1)
        coordinateSum = engine.contract({splitKey: self.coresDict[splitKey]}, openColors=[])[:]

        if coordinateSum == 0:
            return "Inconsistent"

        testCore = 1 / coordinateSum * engine.contract({"core0": core0, "core1": core1},
                                                       openColors=self.coresDict[splitKey].colors)
        if names is None:
            names = [splitKey + "_0", splitKey + "_1"]
        if engine.cores_close(testCore, self.coresDict[splitKey]):
            del self.coresDict[splitKey]
            self.coresDict[names[0]] = 1 / coordinateSum * core0
            self.coresDict[names[1]] = core1
            if self.maintain_indices:
                self.rebuild_indices()
            return True
        else:
            return False

    def replace_by_support(self, coreKey):
        """
        Replaces a core by its support, and erase if it is trivial
        """
        if not has_trivial_support(self.coresDict[coreKey]):
            self.coresDict[coreKey] = support_of(self.coresDict[coreKey])
        else:
            del self.coresDict[coreKey]
        if self.maintain_indices:
            self.rebuild_indices()


def support_of(core):
    return engine.create_from_slice_iterator(
        shape=core.shape[:], colors=core.colors[:],
        sliceIterator=[(1, {color: idx[i] for i, color in enumerate(core.colors)}) for idx in np.ndindex(*core.shape) if
                       core[{color: idx[i] for i, color in enumerate(core.colors)}] != 0]
    )


def has_trivial_support(core):
    return all([core[idx] != 0 for idx in np.ndindex(*core.shape)])


if __name__ == "__main__":
    conNet = ConstraintNetwork(coresDict={"tCore0": engine.create_from_slice_iterator(
        colors=["a", "b"],
        shape=[2, 2],
        sliceIterator=[(1, {"a": 1})]
    )})
    print(conNet.coresDict["tCore0"].values)
    conNet.split_edge("tCore0", colors0=["a"], colors1=["b"])
    assert len(conNet.coresDict) == 2

    core = conNet.coresDict["tCore0_0"]
    print([(1, {color: idx[i] for i, color in enumerate(core.colors)}) for idx in np.ndindex(*core.shape) if
           core[{color: idx[i] for i, color in enumerate(core.colors)}] != 0])
