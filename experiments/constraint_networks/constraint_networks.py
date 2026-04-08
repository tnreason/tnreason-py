from tnreason import engine

class ConstraintNetwork:
    def __init__(self, coresDict):
        self.coresDict = coresDict

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
        if testCore == self.coresDict[splitKey]:
            del self.coresDict[splitKey]
            self.coresDict[names[0]] = 1 / coordinateSum * core0
            self.coresDict[names[1]] = core1
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


import numpy as np

def support_of(core):
    return engine.create_from_slice_iterator(
        shape=core.shape, colors=core.colors,
        sliceIterator=[(1, {color: idx[i] for i, color in enumerate(core.colors)}) for idx in np.ndindex(core.shape) if
                       core[{color: idx[i] for i, color in enumerate(core.colors)}] != 0]
    )

def has_trivial_support(core):
    return all([core[idx] != 0 for idx in np.ndindex(core.shape)])


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
    print([(1, {color: idx[i] for i, color in enumerate(core.colors)}) for idx in np.ndindex(core.shape) if
           core[{color: idx[i] for i, color in enumerate(core.colors)}] != 0])
