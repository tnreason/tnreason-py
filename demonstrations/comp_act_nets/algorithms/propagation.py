from tnreason.engine import contract
from tnreason.engine import create_from_slice_iterator as create

class ContractionPropagation:
    def __init__(self, cores):
        self.cores = cores
        self.directions = {send: [receive for receive in cores if
                                  set(cores[send].colors) & set(
                                      cores[receive].colors) and receive != send]
                           for send in cores}
        self.messages = {receive: {} for receive in self.cores}

    def initial_message(self, send, receive):
        commonColors = list(set(self.cores[send].colors) & set(self.cores[receive].colors))
        shape = [self.cores[send].shape[i]
                 for i, c in enumerate(self.cores[send].colors) if c in commonColors]
        return create(shape=shape, colors=commonColors, sliceIterator=[(1, {})])

    def calculate_message(self, send, receive):
        """
        Contract received messages with hypercore to send new
        """
        return contract({send: self.cores[send],
                         **{preSend: self.messages[send][preSend] for preSend in self.messages[send]
                            if preSend != receive}},
                        openColors=list(set(self.cores[send].colors) & set(self.cores[receive].colors)))

    def tree_propagation(self):
        """
        Tree implementation: Start with leafs and schedule new if all others received
        """
        schedule = [(send, receive) for send in self.cores for receive in
                    self.directions[send] if len(self.directions[send]) == 1]
        while len(schedule) > 0:
            send, receive = schedule.pop()
            self.messages[receive][send] = self.calculate_message(send, receive)
            for next in self.directions[receive]:
                if (not receive in self.messages[next] and
                        all([(otherSendKey in self.messages[receive] or otherSendKey == next or
                              receive not in self.directions[otherSendKey]) for
                             otherSendKey in self.directions])):
                    schedule.append((receive, next))

    def directed_propagation(self, edgeDirections):
        """
        Given the direction of edges in a dictionary, sent only in direction
        """
        filteredDirections = {
            send: [
                receive for receive in self.directions[send]
                if (common := set(self.cores[send].colors) & set(self.cores[receive].colors))
                   and common.issubset(set(edgeDirections[send][1]))
                   and common.issubset(set(edgeDirections[receive][0]))
            ]
            for send in self.directions
        }

        schedule = [(send, receive) for send in filteredDirections for receive in filteredDirections[send] if
                    len(edgeDirections[send][0]) == 0]

        while len(schedule) > 0:
            send, receive = schedule.pop()
            self.messages[receive][send] = self.calculate_message(send, receive)
            for x in set(edgeDirections[send][1]) & set(edgeDirections[receive][0]):
                edgeDirections[receive][0].remove(x)
            if len(edgeDirections[receive][0]) == 0:
                schedule = schedule + [(receive, next) for next in filteredDirections[receive] if (receive, next) not in schedule]

    def constraint_propagation(self, startSendKeys):
        """
        Constraint implementation: Schedule new when message support changed
        """
        schedule = [(send, receive) for send in startSendKeys for receive in
                    self.directions[send]]
        while len(schedule) > 0:
            send, receive = schedule.pop()
            message = (self.messages[receive][send].clone() if send in self.messages[receive]
                       else self.initial_message(send, receive))
            cont = self.calculate_message(send, receive)

            messageChanged = False
            for val, pos in message:
                if message[pos] != 0 and cont[pos] == 0:
                    message[pos] = - message[pos]
                    messageChanged = True
            self.messages[receive][send] = message

            for next in self.directions[receive]:
                if messageChanged and next != receive and (receive, next) not in schedule:
                    schedule.append((receive, next))
