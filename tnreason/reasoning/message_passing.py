from tnreason.reasoning import variational_inference as vi
from sortedcontainers import SortedList


def standard_clusters_from_computationCoreDict(computationCoreDict):
    """
    Returns messageClusters and inferenceClusters from a computationCoreDict, where
    - messageClusters consist in single variables
    - inferenceClusters consist in all variables involved in a computationCore
    Requires the standard featureDict initialization from a computationCoreDict.
    """
    variableColors = []
    for coreKey in computationCoreDict:
        for i, colorKey in enumerate(computationCoreDict[coreKey].colors):
            if colorKey not in variableColors:
                variableColors.append(colorKey)
    return {colorKey: [colorKey] for colorKey in variableColors}, {
            "_".join(computationCoreDict[coreKey].colors[::-1]): computationCoreDict[coreKey].colors + [
                            "_".join(computationCoreDict[coreKey].colors[::-1])] for coreKey in computationCoreDict}


class ForwardMessagePasser(vi.InferenceBase):
    def __init__(self, startMessageSchedule=None,
                 inferenceClusters=dict(),  # Features being inferred together
                 messageClusters=None,  # Features carrying a message together
                 messageArchitecture=None,  # To each message Cluster a list of its inference Clusters
                 forwardInferenceMethod="ForwardContractor",
                 backwardInferenceMethod="BackwardAlternator",
                 allowClearning=False,
                 **inferenceSpec):
        super().__init__(**inferenceSpec)

        self.inferenceClusters = inferenceClusters
        self.messageClusters = self.initialize_messageClusters(messageClusters, inferenceClusters)
        self.messageArchitecture = self.initialize_messageArchitecture(messageArchitecture, inferenceClusters, self.messageClusters)
        self.messages = self.initialize_messages(self.messageClusters, self.messageArchitecture)

        if startMessageSchedule is None:
            self.messageQueue = SortedList()  # FIFO -> Stack
        else:
            self.messageQueue = SortedList(startMessageSchedule)

        self.messageCount = 0
        self.fixedFeatures = []

        self.allowCleaning = allowClearning
        self.cleanQueue = False

        self.forwardInferenceMethod = forwardInferenceMethod
        self.backwardInferenceMethod = backwardInferenceMethod
    
    def update_inferenceClusters(self, newInferenceClusters):
        self.inferenceClusters = newInferenceClusters
        self.messageClusters = self.initialize_messageClusters(self.messageClusters, newInferenceClusters)
        self.messageArchitecture = self.initialize_messageArchitecture(self.messageArchitecture, newInferenceClusters, self.messageClusters)
        self.messages = self.initialize_messages(self.messageClusters, self.messageArchitecture)

    def initialize_messages(self, messageClusters, messageArchitecture):
        # Initialize messages
        messages = {
            messageCluster: {inferenceCluster: dict() for inferenceCluster in messageArchitecture[messageCluster]}
            for
            messageCluster in messageClusters  # dictionary of received messages
        }
        return messages
        
    def initialize_messageArchitecture(self, messageArchitecture, inferenceClusters, messageClusters):
        # Initialize allowed message directions
        # if messageArchitecture is not None:
        #     self.messageArchitecture = messageArchitecture
        if messageArchitecture is None:
            messageArchitecture = {
                messageClusterKey: [inferenceClusterKey for inferenceClusterKey in inferenceClusters if
                                    all([featureKey in inferenceClusters[inferenceClusterKey] for featureKey in
                                         messageClusters[messageClusterKey]])
                                    ] for messageClusterKey in messageClusters}
        return messageArchitecture

    def initialize_messageClusters(self, messageClusters, inferenceClusters):
        # if messageClusters is not None:
        #     self.messageClusters = messageClusters
        if messageClusters is None:  # Initialize via inference cluster intersections
            messageClusters = dict()
            for firstKey in inferenceClusters:
                for secondKey in inferenceClusters:
                    if firstKey != secondKey:
                        intersection = set(inferenceClusters[firstKey]) & set(inferenceClusters[secondKey])
                        if len(intersection) != 0 and intersection not in [set(v) for v in
                                                                           messageClusters.values()]:
                            messageClusters[f"{firstKey}_{secondKey}"] = list(intersection)
        return messageClusters


    def compute_canParam_message(self, inferenceCluster, messageCluster):
        # assert inferenceCluster in self.messageArchitecture[messageCluster], "Cluster {} is not a parent of cluster {}.".format(
        #    inferenceCluster, messageCluster)

        ## Forward inference in sendKeys: Infer using the effective canParams (those assigned and those communicated from other inference clusters)
        forwardInferer = vi.get_inferer(self.forwardInferenceMethod)(
            caNetwork=self.get_caNetwork(featureKeys=self.inferenceClusters[inferenceCluster],
                                         canParamDict={featureKey:
                                             self.caNetwork.featureDict[featureKey].combine_canParams(
                                                 [self.caNetwork.canParamDict[featureKey]] + [
                                                     self.messages[messageCluster][otherInferenceCluster][featureKey]
                                                     for messageCluster in
                                                     self.messages
                                                     for otherInferenceCluster in self.messages[messageCluster]
                                                     if inferenceCluster in self.messageArchitecture[messageCluster]
                                                        and featureKey in self.messages[messageCluster][
                                                            otherInferenceCluster]
                                                        and otherInferenceCluster != inferenceCluster]
                                             )
                                             for featureKey in self.inferenceClusters[inferenceCluster]})
        )
        forwardInferer.infer_meanParams(self.messageClusters[messageCluster])

        # ## Find features which meanParam has changed, could also select based on canonical parameters
        changedFeatures = []
        for featureKey in self.messageClusters[messageCluster]:
            if featureKey in self.meanParamDict:
                if not self.meanParamDict[featureKey] == forwardInferer.meanParamDict[featureKey]:
                    self.meanParamDict[featureKey] = forwardInferer.meanParamDict[featureKey]
                    changedFeatures.append(featureKey)
            else:
                self.meanParamDict[featureKey] = forwardInferer.meanParamDict[featureKey]
                changedFeatures.append(featureKey)

        ## Compute the message to be sent as canParams
        backwardInferer = vi.get_inferer(self.backwardInferenceMethod)(
            caNetwork=self.get_caNetwork(self.messageClusters[messageCluster]),
            meanParamDict={key: forwardInferer.meanParamDict[key] for key in self.messageClusters[messageCluster]},
        )
        backwardInferer.alternating_updates(
            featureKeys={featureKey for featureKey in self.messageClusters[messageCluster] if not
            "passive" in self.caNetwork.featureDict[featureKey].featureProperties},
            sweepNum=1)

        messageCanParamDict = backwardInferer.caNetwork.canParamDict
        for featureKey in self.messageClusters[messageCluster]:
            if "hard" in self.caNetwork.featureDict[featureKey].featureProperties:
                """
                Hard feature canParams used to directly modify the canonical parameters
                -> If support changes, the feature gets assigned to changedMeans
                """
                self.caNetwork.canParamDict[featureKey] = self.caNetwork.featureDict[featureKey].combine_canParams(
                    [self.caNetwork.canParamDict[featureKey], messageCanParamDict[featureKey]]
                )
                checkSum = self.caNetwork.featureDict[featureKey].activation_sum(
                    self.caNetwork.canParamDict[featureKey])
                if checkSum == 1:
                    self.fixedFeatures.append(featureKey)
                    if self.allowCleaning:
                        self.cleanQueue = True
                elif checkSum == 0:
                    raise ValueError(
                        "Hard feature {} received a message that eliminates all support.".format(featureKey))

            elif not "passive" in self.caNetwork.featureDict[featureKey].featureProperties:
                """
                Soft features canParams are communicated by differences to rest messages
                """
                for parentCluster in self.messages[messageCluster]:
                    if parentCluster != inferenceCluster and featureKey in self.messages[messageCluster][parentCluster]:
                        # Then this is a message from receiveCluster to parentCluster
                        messageCanParamDict[featureKey] += -1 * self.messages[messageCluster][parentCluster][
                            featureKey]
                self.messages[messageCluster][inferenceCluster][featureKey] = messageCanParamDict[featureKey]

        self.messageCount += 1
        return changedFeatures

    def propagate_until_convergence(self, nonTrivialFeatureKeys, maxMessageCount=None, verbose=False):
        self.add_affected_directions(nonTrivialFeatureKeys)
        while len(self.messageQueue) > 0 and (maxMessageCount is None or self.messageCount < maxMessageCount):
            sendCluster, receiveCluster = self.messageQueue.pop()
            changedMeans = self.compute_canParam_message(sendCluster, receiveCluster)

            if verbose:
                print("Message {} passed from cluster {} to cluster {}. Changed feature means: {}".format(
                    self.messageCount, sendCluster, receiveCluster, changedMeans))

            self.add_affected_directions([featureKey for featureKey in changedMeans],
                                         exceptionList=[(sendCluster, receiveCluster)], verbose=verbose)
            if self.cleanQueue:
                self.remove_fixed_messages()
        if maxMessageCount:
            print("Message passing terminated after {} of allowed {} messages.".format(self.messageCount,
                                                                                       maxMessageCount))
        else:
            print("Message passing terminated after {} messages.".format(self.messageCount))

    def add_affected_directions(self, featureKeys, exceptionList=[], verbose=False):
        affectedDirections = [(inferenceCluster, messageCluster) for messageCluster in self.messageArchitecture for
                              inferenceCluster in
                              self.messageArchitecture[messageCluster] if
                              any([featureKey in self.inferenceClusters[inferenceCluster] for featureKey in
                                   featureKeys]) and (
                                  inferenceCluster, messageCluster) not in exceptionList]
        if verbose:
            print("Adding affected directions to message queue: {}".format(affectedDirections))
        self.messageQueue.update([cpPair for cpPair in affectedDirections if not cpPair in self.messageQueue])

    def remove_fixed_messages(self):
        self.messageQueue = SortedList([(inferenceCluster, messageCluster) for (inferenceCluster, messageCluster) in
                                        self.messageQueue
                                        if not all(
                [featureKey in self.fixedFeatures for featureKey in self.messageClusters[messageCluster]])])
        self.cleanQueue = False
