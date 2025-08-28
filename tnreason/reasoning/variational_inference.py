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
        if "passive" in self.caNetwork.featureDict[featureKey].featureProperties:
            self.meanParamDict[featureKey] = None
            return None
        else:
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
            if not "passive" in self.caNetwork.featureDict[featureKey].featureProperties:
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

    def __init__(self, clusterDict=dict(), startMessageSchedule=None,
                 forwardInferenceMethod="ForwardContractor",
                 backwardInferenceMethod="BackwardAlternator", **inferenceSpec):
        super().__init__(**inferenceSpec)

        self.forwardInferenceMethod = forwardInferenceMethod
        self.backwardInferenceMethod = backwardInferenceMethod

        self.clusterFeatures = clusterDict  # Dictionary of featureKeys to clusters: send and message clusters!
        self.clusterParents = {messageCluster: [inferenceCluster for inferenceCluster in self.clusterFeatures if
                                          messageCluster != inferenceCluster and  # childKey is not a parent of itself
                                          all([featureKey in self.clusterFeatures[inferenceCluster] for featureKey in
                                               self.clusterFeatures[messageCluster]])] for messageCluster in
                               self.clusterFeatures}  # To each clusterKey the parents

        self.messageDict = {
            messageCluster: {inferenceCluster: dict() for inferenceCluster in self.clusterParents[messageCluster]} for
            messageCluster in self.clusterFeatures  # dictionary of received messages
        }

        if startMessageSchedule is None:
            self.messageQueue = SortedList()  # FIFO -> Stack
        else:
            self.messageQueue = SortedList(startMessageSchedule)

        self.messageCount = 0

    def propagate_until_convergence(self, nonTrivialFeatureKeys, maxMessageCount=None, verbose=False):
        self.add_affected_directions(nonTrivialFeatureKeys)
        while len(self.messageQueue) > 0 and (maxMessageCount is None or self.messageCount < maxMessageCount):
            sendCluster, receiveCluster = self.messageQueue.pop()
            changedMeans = self.compute_canParam_message(sendCluster, receiveCluster)
            self.add_affected_directions([featureKey for featureKey in changedMeans],
                                         exceptionList=[(sendCluster, receiveCluster)])
            if verbose:
                print("Message {} passed from cluster {} to cluster {}. Changed feature means: {}".format(
                    self.messageCount, sendCluster, receiveCluster, changedMeans))
        if maxMessageCount:
            print("Message passing terminated after {} of allowed {} messages.".format(self.messageCount,
                                                                                       maxMessageCount))
        else:
            print("Message passing terminated after {} messages.".format(self.messageCount))

    # def collapse_hard_feature_message(self, hardFeatureKeys):
    #     """
    #     Optional: Smallen the message dictionary by absorbing messages to the canParams
    #     """
    #     for featureKey in hardFeatureKeys:
    #         assert type(self.caNetwork.featureDict[featureKey]) == representation.HardPartitionFeature
    #         self.caNetwork.canParamDict[featureKey] = self.caNetwork.featureDict[featureKey].combine_canParams(
    #             [self.caNetwork.canParamDict[featureKey]] + [
    #                 self.messageDict[messageCluster][inferenceCluster].pop(featureKey) for messageCluster in
    #                 self.messageDict for inferenceCluster in self.messageDict[messageCluster]
    #                 if featureKey in self.clusterFeatures[inferenceCluster]]
    #         )

    def add_affected_directions(self, featureKeys, exceptionList=[]):
        affectedDirections = [(inferenceCluster, messageCluster) for messageCluster in self.clusterParents for inferenceCluster in
                              self.clusterParents[messageCluster] if
                              any([featureKey in self.clusterFeatures[inferenceCluster] for featureKey in featureKeys]) and (
                                  inferenceCluster, messageCluster) not in exceptionList]
        self.messageQueue.update([cpPair for cpPair in affectedDirections if not cpPair in self.messageQueue])

    def compute_canParam_message(self, inferenceCluster, messageCluster):
        assert inferenceCluster in self.clusterParents[messageCluster], "Cluster {} is not a parent of cluster {}.".format(
            inferenceCluster, messageCluster)

        ## Forward inference in sendKeys: Infer using the effective canParams (those assigned and those communicated from other inference clusters)
        forwardInferer = get_inferer(self.forwardInferenceMethod)(
            caNetwork=self.get_caNetwork(self.clusterFeatures[inferenceCluster], canParamDict={featureKey:
                self.caNetwork.featureDict[featureKey].combine_canParams(
                    [self.caNetwork.canParamDict[featureKey]] + [
                        self.messageDict[messageCluster][otherInferenceCluster][featureKey] for messageCluster in
                        self.messageDict
                        for otherInferenceCluster in self.messageDict[messageCluster]
                        if inferenceCluster in self.clusterParents[messageCluster]
                           and featureKey in self.messageDict[messageCluster][otherInferenceCluster]
                           and otherInferenceCluster != inferenceCluster]
                )
                for featureKey in self.clusterFeatures[inferenceCluster]})
        )
        forwardInferer.infer_meanParams(self.clusterFeatures[messageCluster])

        # ## Find features which meanParam has changed, could also select based on canonical parameters
        changedFeatures = []
        for featureKey in self.clusterFeatures[messageCluster]:
            if featureKey in self.meanParamDict:
                if not self.meanParamDict[featureKey] == forwardInferer.meanParamDict[featureKey]:
                    self.meanParamDict[featureKey] = forwardInferer.meanParamDict[featureKey]
                    changedFeatures.append(featureKey)
            else:
                self.meanParamDict[featureKey] = forwardInferer.meanParamDict[featureKey]
                changedFeatures.append(featureKey)

        ## Compute the message to be sent as canParams
        backwardInferer = get_inferer(self.backwardInferenceMethod)(
            caNetwork=self.get_caNetwork(self.clusterFeatures[messageCluster]),
            meanParamDict={key: forwardInferer.meanParamDict[key] for key in self.clusterFeatures[messageCluster]},
        )
        backwardInferer.alternating_updates(
            featureKeys={featureKey for featureKey in self.clusterFeatures[messageCluster] if not
            "passive" in self.caNetwork.featureDict[featureKey].featureProperties},
            sweepNum=1)

        messageCanParamDict = backwardInferer.caNetwork.canParamDict
        for featureKey in self.clusterFeatures[messageCluster]:
            if type(self.caNetwork.featureDict[featureKey]) in [representation.SoftPartitionFeature,
                                                                representation.SingleSoftFeature]:
                """
                Soft features canParams are communicated by differences to rest messages
                """
                for parentCluster in self.messageDict[messageCluster]:
                    if parentCluster != inferenceCluster and featureKey in self.messageDict[messageCluster][parentCluster]:
                        # Then this is a message from receiveCluster to parentCluster
                        messageCanParamDict[featureKey] += -1 * self.messageDict[messageCluster][parentCluster][
                            featureKey]
                self.messageDict[messageCluster][inferenceCluster][featureKey] = messageCanParamDict[featureKey]
            elif type(self.caNetwork.featureDict[featureKey]) in [representation.HardPartitionFeature]:
                """
                Hard feature canParams used to directly modify the canonical parameters
                -> If support changes, the feature gets assigned to changedMeans
                """
                self.caNetwork.canParamDict[featureKey] = self.caNetwork.featureDict[featureKey].combine_canParams(
                    [self.caNetwork.canParamDict[featureKey], messageCanParamDict[featureKey]]
                )

        self.messageCount += 1
        return changedFeatures
