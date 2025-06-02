from tnreason import engine
from tnreason.reasoning import features as ft
import numpy as np

canCorePre = "_can"  # Suffix for canonical parameter cores, to distinguish from the exponentiated ones being propert activation cores


def get_inferer(inferenceMethod):
    if inferenceMethod is None or inferenceMethod == "ForwardContractor":
        return ForwardContractor
    elif inferenceMethod == "BackwardAlternator":
        return BackwardAlternator
    elif inferenceMethod == "ExpectationPropagator":
        return ExpectationPropagator
    else:
        raise ValueError("Inference Method {} not implemented!".format(inferenceMethod))


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

    def get_caNetwork(self, featureKeys, canParamDict=None):
        reducedFeatureDict = dict()
        reducedComputationCoreDict = dict()

        for featureKey in featureKeys:
            reducedFeatureDict[featureKey] = self.caNetwork.featureDict[featureKey]
            reducedComputationCoreDict.update(
                {comCoreKey: self.caNetwork.computationCoreDict[comCoreKey] for comCoreKey in
                 self.caNetwork.featureDict[featureKey].affectedComputationCores}
            )

        if canParamDict is None:
            reducedCanParamDict = {featureKey: self.caNetwork.canParamDict[featureKey] for featureKey in featureKeys}
        else:
            reducedCanParamDict = canParamDict

        return ft.ComputationActivationNetwork(featureDict=reducedFeatureDict,
                                               computationCoreDict=reducedComputationCoreDict,
                                               baseMeasureCoreDict=self.caNetwork.baseMeasureCoreDict,
                                               canParamDict=reducedCanParamDict)


class ForwardContractor(InferenceBase):
    method = "ForwardContractor"

    """
    Calculates mean parameters by direct contraction
    """

    def infer_meanParam(self, featureKey):
        self.meanParamDict[featureKey] = self.caNetwork.featureDict[featureKey].compute_meanParam(
            self.compute_environmentMean(featureKey, normalize=True))
        return self.meanParamDict[featureKey]

    def compute_environmentMean(self, featureKey, normalize=False):
        """
        Need normalization, when contracting for mean Parameters, not for local canParam updates.
        """
        preEnvironmentMean = engine.contract(
            coreDict=self.caNetwork.create_cores(),
            openColors=self.caNetwork.featureDict[featureKey].featureColors,
            dimensionDict=self.dimensionDict
        )
        if normalize:
            return 1 / engine.contract({"pre": preEnvironmentMean}, openColors=[])[:] * preEnvironmentMean
        else:
            return preEnvironmentMean

    def infer_meanParams(self, featureKeys=None):
        if featureKeys is None:
            featureKeys = list(self.caNetwork.featureDict.keys())
        for featureKey in featureKeys:
            self.infer_meanParam(featureKey)


class BackwardAlternator(InferenceBase):
    method = "BackwardAlternator"

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
        environmentMean = self.forwardInferer.compute_environmentMean(featureKey, normalize=False)
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
        weightDict = {featureKey: [] for featureKey in featureKeys}
        for _ in range(sweepNum):
            for featureKey in featureKeys:
                self.update_canParam(featureKey)
                weightDict[featureKey].append(self.caNetwork.canParamDict[featureKey])
        return weightDict


class ExpectationPropagator(InferenceBase):
    method = "ExpectationPropagator"

    """
    Forward inference by propagation of messages (additive canParams) through the computation activation network.
    """

    def __init__(self, clusterDict=dict(), forwardInferenceMethod="ForwardContractor",
                 backwardInferenceMethod="BackwardAlternator", **inferenceSpec):
        super().__init__(**inferenceSpec)
        self.clusterFeatures = clusterDict  # Dictionary of featureKeys to clusters: send and message clusters!
        self.clusterParents = {childKey: [parentKey for parentKey in self.clusterFeatures if
                                          childKey != parentKey and  # childKey is not a parent of itself
                                          all([featureKey in self.clusterFeatures[parentKey] for featureKey in
                                               self.clusterFeatures[childKey]])] for childKey in
                               self.clusterFeatures}  # To each clusterKey the parents

        self.forwardInferenceMethod = forwardInferenceMethod
        self.backwardInferenceMethod = backwardInferenceMethod

        self.messageDict = {
            receiveCluster: dict() for receiveCluster in self.clusterFeatures  # dictionary of received messages
        }

    def compute_canParam_message(self, sendCluster, receiveCluster):
        assert sendCluster in self.clusterParents[receiveCluster], "Cluster {} is not a parent of cluster {}.".format(
            sendCluster, receiveCluster)

        # Compute effective canParams: Those assigned to the sendCluster and those received from childClusters
        effectiveCanParamDict = {featureKey: self.caNetwork.canParamDict[featureKey] for featureKey in
                                 self.clusterFeatures[sendCluster]}
        for childCluster in self.clusterFeatures:
            if childCluster != sendCluster and sendCluster in self.clusterParents[childCluster]:
                # Then childCluster is a child of sendCluster and its other received messages need to be added
                for otherReceiveCluster in self.messageDict[childCluster]:
                    if otherReceiveCluster != sendCluster:
                        # Then this is a message from childCluster to sendCluster
                        for featureKey in self.clusterFeatures[childCluster]:
                            effectiveCanParamDict[featureKey] += self.messageDict[childCluster][otherReceiveCluster][
                                featureKey]

        ## Forward inference in sendKeys
        forwardInferer = get_inferer(self.forwardInferenceMethod)(
            caNetwork=self.get_caNetwork(self.clusterFeatures[sendCluster], canParamDict=effectiveCanParamDict)
        )
        forwardInferer.infer_meanParams(self.clusterFeatures[receiveCluster])

        backwardInferer = get_inferer(self.backwardInferenceMethod)(
            caNetwork=self.get_caNetwork(self.clusterFeatures[receiveCluster]),
            meanParamDict={key: forwardInferer.meanParamDict[key] for key in self.clusterFeatures[receiveCluster]},
        )
        backwardInferer.alternating_updates(featureKeys=self.clusterFeatures[receiveCluster])

        messageCanParamDict = backwardInferer.caNetwork.canParamDict
        for parentCluster in self.messageDict[receiveCluster]:
            if parentCluster != sendCluster:
                # Then this is a message from receiveCluster to parentCluster
                for featureKey in self.clusterFeatures[receiveCluster]:
                    messageCanParamDict[featureKey] += -1 * self.messageDict[receiveCluster][parentCluster][featureKey]

        self.messageDict[receiveCluster][sendCluster] = messageCanParamDict
