import numpy as np

from tnreason import engine


class GenericMeanField:
    def __init__(self, energyDict, colors=[], dimDict={}, edgeColorDict=None):
        """
        Edge Color Dict : Generalizing partition color to arbitrary markov networks
        """

        self.energyDict = energyDict
        self.colors = colors
        self.dimDict = dimDict

        if edgeColorDict is None:
            self.edgeColorDict = {color: [color] for color in colors}
        else:
            self.edgeColorDict = edgeColorDict

        self.approxCores = {parKey: engine.create_trivial_core(parKey, [self.dimDict[color] for color in
                                                                        self.edgeColorDict[parKey]],
                                                               self.edgeColorDict[parKey]).multiply(
            1 / np.prod([self.dimDict[color] for color in self.edgeColorDict[parKey]])) for parKey in
            self.edgeColorDict}

    def update_core(self, approxCoreKey):

        energyKeys = list(self.energyDict.keys())
        contracted = engine.contract(
            {**{key: self.approxCores[key] for key in self.approxCores if key != approxCoreKey},
             **self.energyDict[energyKeys[0]][0]},
            openColors=self.edgeColorDict[approxCoreKey]
        ).multiply(self.energyDict[energyKeys[0]][1])
        for energyKey in energyKeys[1:]:
            contracted = contracted.sum_with(engine.contract(
                {**{key: self.approxCores[key] for key in self.approxCores if key != approxCoreKey},
                 **self.energyDict[energyKey][0]},
                openColors=self.edgeColorDict[approxCoreKey]
            ).multiply(self.energyDict[energyKey][1]))

        ## Entropy Terms
        for secCoreKey in [key for key in self.approxCores if key != approxCoreKey]:
            contracted = contracted.sum_with(engine.contract(
                {**{key: self.approxCores[key] for key in self.approxCores if key != approxCoreKey},
                 "entropyTerm" : self.approxCores[secCoreKey].build_ln()},
                openColors=self.edgeColorDict[approxCoreKey]
            ).multiply(-1))

        ## Normate coordinatewise
        denominator = engine.contract({key: self.approxCores[key] for key in self.approxCores if key != approxCoreKey},
                                      openColors=self.edgeColorDict[approxCoreKey])

        update = engine.create_trivial_core(approxCoreKey,
                                            [self.dimDict[color] for color in self.edgeColorDict[approxCoreKey]],
                                            self.edgeColorDict[approxCoreKey])
        update.values = update.values.astype(float) # Work around to get float NumpyCore
        sum = 0
        for i in np.ndindex(*[self.dimDict[color] for color in self.edgeColorDict[approxCoreKey]]):
            coordinate = np.exp(contracted[i] / denominator[i])
            update[i] = coordinate
            sum = sum + coordinate

        self.approxCores[approxCoreKey] = update.multiply(1/sum)


if __name__ == "__main__":
    from tnreason import engine, knowledge

    energyDict = knowledge.HybridKnowledgeBase(
        weightedFormulas={"w1": ["or", "a", "b", 1],
                          "w2": ["a",1]}
    ).get_energy_dict()

    approximator = GenericMeanField(energyDict, colors=["a", "b"], dimDict={"a": 2, "b": 2}, edgeColorDict={"e1":["a","b"]})
    approximator.update_core("e1")
    print(approximator.approxCores["e1"].values)
