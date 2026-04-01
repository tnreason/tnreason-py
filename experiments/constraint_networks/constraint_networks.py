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
            del self.coresDict[key]

    def split_edge(self, splitKey, colors0, colors1):
        ## Check whether can be splitted
        core0 = engine.contract({splitKey: self.coresDict[splitKey]}, openColors=colors0)
        core1 = engine.contract({splitKey: self.coresDict[splitKey]}, openColors=colors1)
        coordinateSum = engine.contract({splitKey: self.coresDict[splitKey]}, openColors=[])[:]

        testCore = 1 / coordinateSum * engine.contract({"core0": core0, "core1": core1},
                                                       openColors=self.coresDict[splitKey].colors)
        if testCore == self.coresDict[splitKey]:
            del self.coresDict[splitKey]
            self.coresDict[splitKey + "_0"] = core0
            self.coresDict[splitKey + "_1"] = core1
        else:
            print("Edge cannot be splitted")


if __name__ == "__main__":
    conNet = ConstraintNetwork(coresDict={"tCore0": engine.create_from_slice_iterator(
        colors=["a", "b"],
        shape=[2, 2],
        sliceIterator=[(1, {"a": 1})]
    )})
    conNet.split_edge("tCore0", colors0=["a"], colors1=["b"])
    assert len(conNet.coresDict) == 2

    ## Test whether modus ponens works, i.e .implication "a \Rightarrow b" disentangled by knowing the premise "a"
    modusPonensNet = ConstraintNetwork(coresDict={
        "impCore": engine.create_from_slice_iterator(
            colors=["a", "b"],
            shape=[2, 2],
            sliceIterator=[(1, dict()), (-1, {"a": 1, "b": 0})]
        ),
        "aKnownCore": engine.create_from_slice_iterator(
            colors=["a"],
            shape=[2],
            sliceIterator=[(1, {"a": 1})]
        )
    }
    )
    modusPonensNet.add_subcontraction(contractKeys=["impCore", "aKnownCore"], openColors=["a", "b"],
                                      dropKeys=["aKnownCore", "impCore"], contractedName="sumCore")
    assert len(modusPonensNet.coresDict) == 1
    modusPonensNet.split_edge("sumCore", colors0=["a"], colors1=["b"])
    assert modusPonensNet.coresDict["sumCore_0"][{"a": 0}] == 0
    assert modusPonensNet.coresDict["sumCore_0"][{"a": 1}] == 1
    assert modusPonensNet.coresDict["sumCore_1"][{"b": 0}] == 0
    assert modusPonensNet.coresDict["sumCore_1"][{"b": 1}] == 1
