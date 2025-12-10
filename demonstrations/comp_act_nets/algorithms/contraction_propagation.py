from tnreason import engine


class ContractionPropagation:
    def __init__(self, tensorNetwork, startEdges=None):
        self.tensorNetwork = tensorNetwork
        if startEdges is None:
            startEdges = tensorNetwork.keys()
        self.messageDirections = get_message_directions(tensorNetwork)
        self.messageSchedule = [(sendKey,receiveKey) for sendKey in startEdges for receiveKey in self.messageDirections[sendKey]]
        self.messageDict = {receiveKey: {} for receiveKey in self.tensorNetwork}

    def calculate_message(self, sendKey, receiveKey):
        ## Contract all received messaged but that from receiving core with the sending hypercore to compute the message
        self.messageDict[receiveKey][sendKey] = engine.contract({
            sendKey: self.tensorNetwork[sendKey],
            **{secSendKey : self.messageDict[sendKey][secSendKey] for secSendKey in self.messageDict[sendKey] if secSendKey != receiveKey}},
            openColors=list(set(self.tensorNetwork[sendKey].colors) & set(self.tensorNetwork[receiveKey].colors)))

    def tree_belief_propagation(self):
        while len(self.messageSchedule) > 0:
            sendKey, receiveKey = self.messageSchedule.pop()
            self.calculate_message(sendKey, receiveKey)
            ## Load all directions for which all other directions are already received
            for nextKey in self.messageDirections[receiveKey]:
                if not receiveKey in self.messageDict[nextKey]:  # when message not yet sent
                    if all([(otherSendKey in self.messageDict[
                        receiveKey] or otherSendKey == nextKey or receiveKey not in self.messageDirections[
                                 otherSendKey]) for otherSendKey in self.messageDirections]): # whether all messages received
                        self.messageSchedule.append((receiveKey, nextKey))

    def constraint_propagation(self):
        while len(self.messageSchedule) > 0:
            sendKey, receiveKey = self.messageSchedule.pop()
            if sendKey in self.messageDict[receiveKey]:
                oldMessage = self.messageDict[receiveKey][sendKey]
            else:  ## Trivial message
                oldMessage = engine.create_from_slice_iterator(
                    shape=[color for color in self.tensorNetwork[sendKey].colors if
                           color in self.tensorNetwork[receiveKey].colors],
                    colors=[color for color in self.tensorNetwork[sendKey].colors if
                            color in self.tensorNetwork[receiveKey].colors],
                    sliceIterator=[(1, {})]
                )
            self.calculate_message(sendKey, receiveKey)
            changed = False
            for val, posDict in oldMessage:
                if oldMessage[posDict] > 0 and self.messageDict[receiveKey][sendKey] == 0:
                    changed = True
            if changed:
                for nextKey in self.messageDict[receiveKey]:
                    if (receiveKey, nextKey) not in self.messageDict[sendKey]:
                        self.messageSchedule.append((sendKey, nextKey))


def get_message_directions(tensorNetwork):
    return {sendKey: [receiveKey for receiveKey in tensorNetwork if
                      set(tensorNetwork[sendKey].colors) & set(tensorNetwork[receiveKey].colors) and receiveKey != sendKey]
            for sendKey in tensorNetwork}
