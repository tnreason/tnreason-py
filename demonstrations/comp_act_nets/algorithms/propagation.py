from tnreason import engine


class ContractionPropagation:
    def __init__(self, tensorNetwork):
        self.tensorNetwork = tensorNetwork
        self.messageDirections = get_message_directions(tensorNetwork)
        # self.messageSchedule = [(sendKey, receiveKey) for sendKey in startEdges for receiveKey in
        #                        self.messageDirections[sendKey]]
        self.messageDict = {receiveKey: {} for receiveKey in self.tensorNetwork}

    def calculate_message(self, sendKey, receiveKey):
        ## Contract all received messaged but that from receiving core with the sending hypercore to compute the message
        self.messageDict[receiveKey][sendKey] = engine.contract({
            sendKey: self.tensorNetwork[sendKey],
            **{secSendKey: self.messageDict[sendKey][secSendKey] for secSendKey in self.messageDict[sendKey] if
               secSendKey != receiveKey}},
            openColors=list(set(self.tensorNetwork[sendKey].colors) & set(self.tensorNetwork[receiveKey].colors)))

    def tree_propagation(self):
        """
        Used the implementation based on the tree-scheduler:
        - Start with the leaf edges
        - Schedule new messages when message not yet sent and all other messages received
        """
        messageSchedule = [(sendKey, receiveKey) for sendKey in self.tensorNetwork for receiveKey in
                           self.messageDirections[sendKey] if len(self.messageDirections[sendKey]) == 1]
        while len(messageSchedule) > 0:
            sendKey, receiveKey = messageSchedule.pop()
            self.calculate_message(sendKey, receiveKey)
            ## Load all directions for which all other directions are already received
            for nextKey in self.messageDirections[receiveKey]:
                if not receiveKey in self.messageDict[nextKey]:  # when message not yet sent
                    if all([(otherSendKey in self.messageDict[
                        receiveKey] or otherSendKey == nextKey or receiveKey not in self.messageDirections[
                                 otherSendKey]) for otherSendKey in
                            self.messageDirections]):  # whether all messages received
                        messageSchedule.append((receiveKey, nextKey))

    def constraint_propagation(self, startSendKeys=None):
        """
        Uses the implementation based on the constraint-scheduler:
        - Start with some specified send edges
        - Schedule all outgoing messages if the support of the incoming has changed
        """
        if startSendKeys is None:
            startSendKeys = self.tensorNetwork.keys()
        messageSchedule = [(sendKey, receiveKey) for sendKey in startSendKeys for receiveKey in
                           self.messageDirections[sendKey] if receiveKey != sendKey]
        mCount = 0
        while len(messageSchedule) > 0:
            sendKey, receiveKey = messageSchedule.pop()
            if sendKey in self.messageDict[receiveKey]:
                oldMessage = self.messageDict[receiveKey][sendKey].clone()
            else:  ## Then intitialize by trivial tensor
                oldMessage = engine.create_from_slice_iterator(
                    shape=[self.tensorNetwork[sendKey].shape[i] for i, color in
                           enumerate(self.tensorNetwork[sendKey].colors) if
                           color in self.tensorNetwork[receiveKey].colors],
                    colors=[color for color in self.tensorNetwork[sendKey].colors if
                            color in self.tensorNetwork[receiveKey].colors],
                    sliceIterator=[(1, {})]
                )

            ## Check whether message has smaller support than
            changed = False
            self.calculate_message(sendKey, receiveKey)
            for val, posDict in oldMessage:
                if oldMessage[posDict] != 0 and self.messageDict[receiveKey][sendKey][posDict] == 0:
                    oldMessage[posDict] = - oldMessage[posDict]
                    changed = True
            self.messageDict[receiveKey][sendKey] = oldMessage

            ## Schedule new messages
            if changed:
                for nextKey in self.messageDirections[receiveKey]:
                    if (receiveKey, nextKey) not in messageSchedule and nextKey != receiveKey:
                        messageSchedule.append((receiveKey, nextKey))

            mCount += 1
        print("MessageCount: {}".format(mCount))


def get_message_directions(tensorNetwork):
    return {sendKey: [receiveKey for receiveKey in tensorNetwork if
                      set(tensorNetwork[sendKey].colors) & set(
                          tensorNetwork[receiveKey].colors) and receiveKey != sendKey]
            for sendKey in tensorNetwork}
