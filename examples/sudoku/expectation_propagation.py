from tnreason import representation, reasoning

from examples.sudoku import representation as rep


def get_featureDict_computationCoresDict_clusterDict(num):
    constraints = rep.get_sudoku_constraints(num)

    atomVariables = ["a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n)
                     for r1 in range(num) for r2 in range(num) for c1 in range(num) for c2 in range(num) for n in
                     range(num ** 2)]

    featureDict = {
        **{atomKey: reasoning.HardPartitionFeature(featureColors=[atomKey], affectedComputationCores={}) for atomKey in
           atomVariables},  # Atomic Variable Features
        **{categoricalKey: reasoning.HardPartitionFeature(featureColors=[categoricalKey], affectedComputationCores={},
                                                          interpretationDict={categoricalKey: range(num ** 2)})
           for categoricalKey in constraints},
        **{categoricalKey + "_" + atomKey: reasoning.HardPartitionFeature(featureColors=[categoricalKey, atomKey],
                                                                          interpretationDict={
                                                                              categoricalKey: range(num ** 2),
                                                                              atomKey: range(2)},
                                                                          affectedComputationCores=[
                                                                              categoricalKey + "_" + atomKey + representation.suf.atoCoreSuf])
           for categoricalKey in constraints for atomKey in constraints[categoricalKey]}
    }

    computationCores = representation.create_categorical_cores(constraints)

    clusterDict = {
        **{atomKey: [atomKey] for atomKey in atomVariables},
        **{categoricalKey: [categoricalKey] for categoricalKey in constraints},
        **{categoricalKey + "_" + atomKey: [categoricalKey, atomKey, categoricalKey + "_" + atomKey]
           for categoricalKey in constraints for atomKey in constraints[categoricalKey]}
    }
    return featureDict, computationCores, clusterDict


## Known assignments into canParams and extraction
def evidence_into_canParams(evidenceDict):
    """
    use, that evidenceKey is a colorKey and a featureKey
    """
    return {evidenceKey: representation.create_basis_core(name=evidenceKey, shape=[2], colors=[evidenceKey],
                                                          numberTuple=[evidenceDict[evidenceKey]]) for evidenceKey in
            evidenceDict}


def canParams_into_evidence(meanParamsDict):
    """
    Extraction so far only from atom features, categorical features would require basis vector checks
    """
    return {featureKey: 1 for featureKey in meanParamsDict if
            meanParamsDict[featureKey][{featureKey: 0}] == 0 and featureKey.startswith("a")}


if __name__ == "__main__":
    num = 2
    verbose = False

    evidenceDict = {
        "a_0_1_0_0_1": 1,
        "a_0_0_0_1_0": 1,
        "a_0_1_1_0_3": 1,
        "a_1_0_0_0_2": 1,
        "a_0_0_1_0_2": 1
    }

    featureDict, computationCores, clusterDict = get_featureDict_computationCoresDict_clusterDict(num=num)
    caNetwork = reasoning.ComputationActivationNetwork(
        featureDict=featureDict,
        computationCoreDict=computationCores,
        canParamDict=evidence_into_canParams(evidenceDict)
    )

    propagator = reasoning.get_inferer("ExpectationPropagator")(
        caNetwork=caNetwork,
        clusterDict=clusterDict,
        meanParamDict=dict()
    )

    from sortedcontainers import SortedList

    messageQueue = SortedList()
    for childKey in propagator.clusterParents:
        for parentKey in propagator.clusterParents[childKey]:
            messageQueue.add((parentKey, childKey))

    while len(messageQueue) > 0:
        parentKey, childKey = messageQueue.pop()
        changedMeans = propagator.compute_canParam_message(parentKey, childKey)

        ## Add further messages to messageQue
        for featureKey in changedMeans:
            if changedMeans[featureKey]:
                ## Add all parent - child messages, when the child contains the feature
                for childKey in propagator.clusterParents:
                    for parentKey in propagator.clusterParents[childKey]:
                        if featureKey in propagator.clusterFeatures[parentKey]:
                            ## Then parentKey, childKey should be in messageQueue: Add if not there
                            if (parentKey, childKey) not in messageQueue:
                                ## Only add if not already in queue
                                messageQueue.add((parentKey, childKey))
                                if verbose:
                                    print("Added message from {} to {}".format(parentKey, childKey))

    solution_array = rep.evidenceDict_to_array(canParams_into_evidence(propagator.meanParamDict), num=num)
    assert solution_array[0, 0] == 4
    assert solution_array[0, 1] == 1
    assert solution_array[1, 0] == 2
    assert solution_array[1, 1] == 3
    assert solution_array[0, 2] == 3
    assert solution_array[0, 3] == 2
    assert solution_array[1, 2] == 4

    from examples.sudoku import visualization as vis

    vis.visualize_sudoku(solution_array.astype(int), number=2)
