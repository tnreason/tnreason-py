from tnreason import engine

from tnreason.reasoning import sampling_base as sh


class ForwardSampleCore(sh.SampleCoreBase):
    """
    Iteratable SampleCore, can be converted into a contractable TensorCore
    """

    def __init__(self, coreDict, **samplingSpec):
        super().__init__(**samplingSpec)
        self.coreDict = coreDict

    def draw_sample(self, startAssignment=dict()):
        """
        Ignores the startAssignment, since drawing from full distribution!
        """
        sample = {}
        for sampleColor in self.colors:
            condProb = engine.contract(self.coreDict,
                                       openColors=[sampleColor],
                                       colorEvidenceDict=sample,
                                       dimensionDict=self.dimensionDict,
                                       contractionMethod=self.contractionMethod,
                                       coreType=self.coreType)
            ## Might add other random engines than numpy!
            sample[sampleColor] = engine.convert(condProb, "NumpyCore").draw_sample(asEnergy=False)[sampleColor]
        return sample
