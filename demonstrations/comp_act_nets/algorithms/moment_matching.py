from tnreason import engine
import math


class MomentMatcher:
    def __init__(self, computationNetwork, headColors, satisfactionDict):
        self.computationNetwork = computationNetwork
        self.headColors = headColors
        self.satisfactionDict = satisfactionDict

        self.hardParameters = {headColor: int(satisfactionDict[headColor]) for headColor in self.headColors if
                               satisfactionDict[headColor] in [0, 1]}
        self.canonicalParameters = {headColor : 0 for headColor in self.headColors if headColor not in self.hardParameters}

    def update_canonical_parameter(self, headColor):
        currentMean = engine.contract({**self.computationNetwork, **create_hard_activation(self.hardParameters),
                                       **create_soft_activation({otherHead: self.canonicalParameters[otherHead]
                                                                 for otherHead in self.canonicalParameters if
                                                                 otherHead != headColor})}, openColors=[headColor])
        self.canonicalParameters[headColor] = math.log(self.satisfactionDict[headColor] * currentMean[{headColor: 0}] / (
                    (1 - self.satisfactionDict[headColor]) * currentMean[{headColor: 1}]))

    def alternate(self, iterations=10):
        for _ in range(iterations):
            for headColor in self.canonicalParameters:
                self.update_canonical_parameter(headColor)

def create_hard_activation(hardParameters):
    return {headColor + "_aC" : engine.create_from_slice_iterator(shape=[2], colors=[headColor],
                                              sliceIterator=[(1, {headColor: hardParameters[headColor]})]) for headColor
            in hardParameters}


def create_soft_activation(canonicalParameters):
    return {headColor + "_aC" : engine.create_from_slice_iterator(shape=[2], colors=[headColor],
                                              sliceIterator=[(1, {headColor: 0}),
                                                             (math.exp(canonicalParameters[headColor]), {headColor: 1})])
            for headColor in canonicalParameters}
