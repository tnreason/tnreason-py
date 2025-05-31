from tnreason import engine

import numpy as np

canCorePre = "_can"  # Suffix for canonical parameter cores, to distinguish from the exponentiated ones being propert activation cores


class InferenceBase(engine.EngineUser):
    """
    Inference on Generic Exponential Distribution
    """

    def __init__(self, caNetwork=None,
                 meanParamDict=dict(),
                 **engineSpec):
        super().__init__(**engineSpec)

        self.caNetwork = caNetwork  # Instance of tnreason.reasoning.features.ComputationActivationNetwork
        self.meanParamDict = meanParamDict


class ForwardContractor(InferenceBase):
    """
    Calculates mean parameters by direct contraction
    """

    def infer_meanParam(self, featureKey):
        self.meanParamDict[featureKey] = self.caNetwork.featureDict[featureKey].compute_meanParam(
            self.compute_environmentMean(featureKey))
        return self.meanParamDict[featureKey]

    def compute_environmentMean(self, featureKey):
        return engine.contract(
            coreDict=self.caNetwork.create_cores(),
            openColors=self.caNetwork.featureDict[featureKey].featureColors,
            dimensionDict=self.dimensionDict
        )

    def infer_meanParams(self, featureKeys=None):
        if featureKeys is None:
            featureKeys = list(self.caNetwork.featureDict.keys())
        for featureKey in featureKeys:
            self.infer_meanParam(featureKey)


class BackwardAlternator(InferenceBase):

    def __init__(self,
                 forwardInferer=None, **inferenceSpec):
        super().__init__(**inferenceSpec)
        if forwardInferer is None:
            """
            By default use the contraction based forward inferer
            """
            self.forwardInferer = ForwardContractor(**inferenceSpec)
        elif not isinstance(forwardInferer, ForwardContractor):
            raise ValueError("forwardInferer needs to be an instance of ForwardContractor.")
        else:
            self.forwardInferer = forwardInferer

    def update_canParam(self, featureKey):
        environmentMean = self.forwardInferer.compute_environmentMean(featureKey)
        updatedCanParam = self.caNetwork.featureDict[featureKey].local_adjustment(
            environmentMean=environmentMean, meanParam=self.meanParamDict[featureKey],
            oldCanParam=self.forwardInferer.caNetwork.canParamDict[featureKey])

        self.caNetwork.canParamDict[featureKey] = updatedCanParam
        self.forwardInferer.caNetwork.canParamDict[featureKey] = updatedCanParam
        return updatedCanParam

    def alternating_updates(self, featureKeys=None, sweepNum=10):
        """
        Alternating updates of the canonical parameters and mean parameters.
        """
        if featureKeys is None:
            featureKeys = list(self.caNetwork.featureDict.keys())
        weightDict = {featureKey : [] for featureKey in featureKeys}
        for _ in range(sweepNum):
            for featureKey in featureKeys:
                self.update_canParam(featureKey)
                weightDict[featureKey].append(self.caNetwork.canParamDict[featureKey])
        return weightDict