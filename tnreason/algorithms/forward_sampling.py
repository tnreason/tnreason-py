from tnreason import engine


class ForwardSampleCore:
    """
    Iteratable SampleCore, can be converted into a contractable TensorCore
    """
    def __init__(self, dist, dimDict=None, coreType=None, contractionMethod=None, sampleNum = 1, sampleColors=None):
        self.distribution = dist
        if dimDict is None:
            self.dimDict = engine.get_dimDict(self.distribution.create_cores())
        else:
            self.dimDict = dimDict
        self.coreType = coreType
        self.contractionMethod = contractionMethod

        self.index = 0
        self.sampleNum = sampleNum

        if sampleColors is None:
            self.colors = list(self.dimDict.keys())
        else:
            self.colors = sampleColors

            for color in self.colors:
                if color not in self.dimDict:
                    self.dimDict[color] = 2

        self.shape = [self.dimDict[color] for color in self.colors]

    def draw_forward_sample(self):
        for color in self.colors:
            if color not in self.dimDict:
                self.dimDict[color] = 2
        sample = {}
        for sampleColor in self.colors:
            condProb = engine.contract(
                {**self.distribution.create_cores(),
                 **{oldColor + "_sampleCore": engine.create_basis_core(oldColor + "_sampleCore",
                                                                       [self.dimDict[oldColor]], [oldColor],
                                                                       sample[oldColor], coreType=self.coreType) for
                    oldColor in sample}},
                openColors=[sampleColor], dimDict=self.dimDict, method=self.contractionMethod)
            sample[sampleColor] = condProb.draw_sample(asEnergy=False)[sampleColor]
        return sample

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.sampleNum:
            self.index += 1
            return (1, self.draw_forward_sample())
        else:
            self.index = 0
            raise StopIteration