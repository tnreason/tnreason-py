from demonstrations.sudoku.constraints import sudoku_constraints as rep

from tnreason import reasoning, engine


def get_forward_mp_inferer(num, startAssignment):
    constraints = rep.get_sudoku_constraints(num)
    atomVariables = ["a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n)
                     for r1 in range(num) for r2 in range(num) for c1 in range(num) for c2 in range(num) for n in
                     range(num ** 2)]

    caNetwork = rep.get_assignment_as_CANetwork(num=num, startAssignment=startAssignment)
    messageClusters, inferenceClusters = reasoning.standard_clusters_from_computationCoreDict(
        caNetwork.computationCoreDict)

    return reasoning.ForwardMessagePasser(
        caNetwork = caNetwork,
        messageClusters=messageClusters,
        inferenceClusters = inferenceClusters,
        meanParamDict={
            **{atomKey: engine.create_from_slice_iterator(
                shape=[2], colors=[atomKey],
                sliceIterator=[(1, {})]) for atomKey in atomVariables},
            **{categoricalKey: engine.create_from_slice_iterator(
                shape=[num ** 2], colors=[categoricalKey],
                sliceIterator=[(1, {})]) for categoricalKey in constraints}},
        allowClearning=False  # aggressively drop settled features to help larger boards converge
    )


def get_simple_clusterDict(num):
    """
    Prepares a simple clusterDict for Expectation-Propagation, where each cluster contains a single atom and categorical
    """
    constraints = rep.get_sudoku_constraints(num)
    atomVariables = ["a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n)
                     for r1 in range(num) for r2 in range(num) for c1 in range(num) for c2 in range(num) for n in
                     range(num ** 2)]

    return {
        **{atomKey: [atomKey] for atomKey in atomVariables},
        **{categoricalKey: [categoricalKey] for categoricalKey in constraints},
        **{categoricalKey + "_" + atomKey: [categoricalKey, atomKey, categoricalKey + "_" + atomKey]
           for categoricalKey in constraints for atomKey in constraints[categoricalKey]}
    }


def meanParams_to_evidence(meanParamsDict):
    """
    Extraction so far only from atom features, categorical features would require basis vector checks
    """
    return {featureKey: 1 for featureKey in meanParamsDict if
            meanParamsDict[featureKey][{featureKey: 0}] == 0 and featureKey.startswith("a")}
