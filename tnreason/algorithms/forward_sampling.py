from tnreason import engine

from tnreason.algorithms import sampling_handling as sh


class ForwardSampleCore(sh.SampleCoreBase):
    """
    Iteratable SampleCore, can be converted into a contractable TensorCore
    """

    def __init__(self, dist, dimDictFromCores=True, **samplingSpec):
        super().__init__(**samplingSpec)
        self.distribution = dist

        if dimDictFromCores:
            self.dimensionDict.update(engine.get_dimDict(self.distribution.create_cores()))

    def draw_sample(self, startAssignment=dict()):
        """
        Ignores the startAssignment, since drawing from full distribution!
        """
        for color in self.colors:
            if color not in self.dimensionDict:
                self.dimensionDict[color] = 2
        sample = {}
        for sampleColor in self.colors:
            condProb = engine.contract(
                {**self.distribution.create_cores(),
                 **{oldColor + "_sampleCore": engine.create_basis_core(oldColor + "_sampleCore",
                                                                       [self.dimensionDict[oldColor]], [oldColor],
                                                                       sample[oldColor], coreType=self.coreType) for
                    oldColor in sample}},
                openColors=[sampleColor], dimDict=self.dimensionDict, contractionMethod=self.contractionMethod,
                coreType=self.coreType)
            ## Might add other random engines than numpy!
            sample[sampleColor] = engine.convert(condProb, "NumpyCore").draw_sample(asEnergy=False)[sampleColor]
        return sample
