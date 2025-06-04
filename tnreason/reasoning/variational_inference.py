from tnreason import engine
from tnreason import representation

from sortedcontainers import SortedList

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

        return representation.ComputationActivationNetwork(featureDict=reducedFeatureDict,
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
            norm = engine.contract({"pre": preEnvironmentMean}, openColors=[])[:]
            assert norm > 0, "Inconsistency detected at environment mean to feature {}".format(featureKey)
            return 1 / norm * preEnvironmentMean
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

    def __init__(self, clusterDict=dict(), startMessageSchedule=None, forwardInferenceMethod="ForwardContractor",
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
            receiveCluster: {sendCluster: dict() for sendCluster in self.clusterParents[receiveCluster]} for
            receiveCluster in self.clusterFeatures  # dictionary of received messages
        }

        if startMessageSchedule is None:
            self.messageQueue = SortedList() # FIFO -> Stack
        else:
            self.messageQueue = SortedList(startMessageSchedule)

        self.messageCount = 0

    def propagate_until_convergence(self, nonTrivialFeatureKeys, maxMessageCount=None, verbose=True):
        self.add_affected_directions(nonTrivialFeatureKeys)
        while len(self.messageQueue) > 0 and (maxMessageCount is None or self.messageCount < maxMessageCount):
            sendCluster, receiveCluster = self.messageQueue.pop()
            changedMeans = self.compute_canParam_message(sendCluster, receiveCluster)
            self.add_affected_directions([featureKey for featureKey in changedMeans],
                                         exceptionList=[(sendCluster, receiveCluster)])
        if verbose:
            print("Message passing terminated after {} of allowed {} messages.".format(self.messageCount,
                                                                                       maxMessageCount))

    def collapse_hard_feature_message(self, hardFeatureKeys):
        """
        Optional: Smallen the message dictionary by absorbing messages to the canParams
        """
        for featureKey in hardFeatureKeys:
            assert type(self.caNetwork.featureDict[featureKey]) == representation.HardPartitionFeature
            self.caNetwork.canParamDict[featureKey] = self.caNetwork.featureDict[featureKey].combine_canParams(
                [self.caNetwork.canParamDict[featureKey]] + [
                    self.messageDict[childCluster][parentCluster].pop(featureKey) for childCluster in
                    self.messageDict for parentCluster in self.messageDict[childCluster]
                    if featureKey in self.clusterFeatures[parentCluster]]
            )

    def add_affected_directions(self, featureKeys, exceptionList=[]):
        affectedDirections = [(parentKey, childKey) for childKey in self.clusterParents for parentKey in
                              self.clusterParents[childKey] if
                              any([featureKey in self.clusterFeatures[parentKey] for featureKey in featureKeys]) and (
                                  parentKey, childKey) not in exceptionList]
        self.messageQueue.update([cpPair for cpPair in affectedDirections if not cpPair in self.messageQueue])

    def compute_canParam_message(self, sendCluster, receiveCluster):
        assert sendCluster in self.clusterParents[receiveCluster], "Cluster {} is not a parent of cluster {}.".format(
            sendCluster, receiveCluster)

        ## Forward inference in sendKeys: Infer using the effective canParams (those assigned and those communicated)
        forwardInferer = get_inferer(self.forwardInferenceMethod)(
            caNetwork=self.get_caNetwork(self.clusterFeatures[sendCluster], canParamDict={featureKey:
                self.caNetwork.featureDict[featureKey].combine_canParams(
                    [self.caNetwork.canParamDict[featureKey]] + [
                        self.messageDict[childCluster][otherParentCluster][featureKey] for childCluster in
                        self.messageDict
                        for otherParentCluster in self.messageDict[childCluster]
                        if sendCluster in self.clusterParents[childCluster]
                           and featureKey in self.messageDict[childCluster][otherParentCluster]
                           and otherParentCluster != sendCluster]
                )
                for featureKey in self.clusterFeatures[sendCluster]})
        )
        forwardInferer.infer_meanParams(self.clusterFeatures[receiveCluster])

        # ## Find features which meanParam has changed
        changedMeans = []
        for featureKey in self.clusterFeatures[receiveCluster]:
            if featureKey in self.meanParamDict:
                if not self.meanParamDict[featureKey] == forwardInferer.meanParamDict[featureKey]:
                    self.meanParamDict[featureKey] = forwardInferer.meanParamDict[featureKey]
                    changedMeans.append(featureKey)
            else:
                self.meanParamDict[featureKey] = forwardInferer.meanParamDict[featureKey]
                changedMeans.append(featureKey)

        ## Compute the message to be sent as canParams
        backwardInferer = get_inferer(self.backwardInferenceMethod)(
            caNetwork=self.get_caNetwork(self.clusterFeatures[receiveCluster]),
            meanParamDict={key: forwardInferer.meanParamDict[key] for key in self.clusterFeatures[receiveCluster]},
        )
        backwardInferer.alternating_updates(featureKeys=self.clusterFeatures[receiveCluster])

        messageCanParamDict = backwardInferer.caNetwork.canParamDict
        for featureKey in self.clusterFeatures[receiveCluster]:
            if type(self.caNetwork.featureDict[featureKey]) in [representation.SoftPartitionFeature, representation.SingleSoftFeature]:
                """
                Soft features canParams are communicated by differences to rest messages
                """
                for parentCluster in self.messageDict[receiveCluster]:
                    if parentCluster != sendCluster and featureKey in self.messageDict[receiveCluster][parentCluster]:
                        # Then this is a message from receiveCluster to parentCluster
                        messageCanParamDict[featureKey] += -1 * self.messageDict[receiveCluster][parentCluster][
                            featureKey]
                self.messageDict[receiveCluster][sendCluster][featureKey] = messageCanParamDict[featureKey]
            elif type(self.caNetwork.featureDict[featureKey]) in [representation.HardPartitionFeature]:
                """
                Hard feature canParams used to directly modify the canonical parameters
                -> If support changes, the feature gets assigned to changedMeans
                """
                self.caNetwork.canParamDict[featureKey] = self.caNetwork.featureDict[featureKey].combine_canParams(
                    [self.caNetwork.canParamDict[featureKey], messageCanParamDict[featureKey]]
                )

        self.messageCount += 1
        return changedMeans
