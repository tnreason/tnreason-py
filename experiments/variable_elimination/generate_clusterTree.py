def generateClusterTree(hyperedgeDict, variableEliminationList, clusterSuf = "_cluster"):

    """
    Generalization to tensor network contraction:
        Hyperedge -> coreKey
        variableKey -> featureKey
        varClusterKey = variableKey + clusterSuf -> cluster generated to variableKey
    """

    clusterHyperedgeDict = {variableKey + clusterSuf: [] for variableKey in variableEliminationList}
    clusterVariablesDict = {variableKey + clusterSuf: set() for variableKey in variableEliminationList}

    assignedHyperEdges = []
    openClusters   = []

    successorDict = {variableKey + clusterSuf : [] for variableKey in variableEliminationList}
    precessorDict = {variableKey + clusterSuf: [] for variableKey in variableEliminationList}

    for variableKey in variableEliminationList:
        varClusterKey = variableKey + clusterSuf

        for hyperKey in hyperedgeDict:
            if hyperKey not in assignedHyperEdges:
                if variableKey in hyperedgeDict[hyperKey]:
                    clusterHyperedgeDict[varClusterKey].append(hyperKey)
                    clusterVariablesDict[varClusterKey] = clusterVariablesDict[variableKey+clusterSuf] | set(hyperedgeDict[hyperKey])
                    assignedHyperEdges.append(hyperKey)

        for openClusterKey in openClusters:
            if variableKey in clusterVariablesDict[openClusterKey]:
                successorDict[openClusterKey].append(varClusterKey)
                precessorDict[varClusterKey].append(openClusterKey)

                clusterVariablesDict[varClusterKey] = clusterVariablesDict[varClusterKey] | clusterVariablesDict[openClusterKey]

                if clusterSuf == "":
                    openClusterEliminatedVariable = openClusterKey
                else:
                    openClusterEliminatedVariable = openClusterKey[:-len(clusterSuf)]
                clusterVariablesDict[varClusterKey].remove(openClusterEliminatedVariable)  # Remove predecessor variable from variable cluster

        for pre in precessorDict[varClusterKey]:
            openClusters.remove(pre) # To not jump over a potentialPre when deleting while iterating in the above loop

        openClusters.append(varClusterKey)

    return clusterHyperedgeDict, clusterVariablesDict, successorDict, precessorDict

def find_message_schedule_towards_root(successorDict, precessorDict):
    messageTuples = []
    while any([len(successorDict[variableClusterKey]) > 0 for variableClusterKey in precessorDict]):
        for variableClusterKey in list(successorDict):
            if len(precessorDict[variableClusterKey]) == 0:
                for nextVariable in successorDict[variableClusterKey]:
                    messageTuples.append((variableClusterKey, nextVariable))
                for nextVariable in successorDict[variableClusterKey]:
                    precessorDict[nextVariable].remove(variableClusterKey)
                successorDict[variableClusterKey] = []
    return messageTuples

if __name__ == "__main__":
    clusterSuf = "_fun"

    hyperedgeDict = {
        "e1": ["a", "b"],
        "e2": ["a"],
        "e3": ["b"],
        "e4": ["b", "c"],
        "e5": ["c"],
    }
    variableEliminationList = ["a", "c", "b"]

    clusterHyperedgeDict, clusterVariablesDict, successorDict, precessorDict = generateClusterTree(hyperedgeDict, variableEliminationList, clusterSuf=clusterSuf)

    print("Cluster Hyperedge Dict:", clusterHyperedgeDict)
    print("Cluster Variables Dict:", clusterVariablesDict)
    print("Successor Dict:", successorDict)
    print("Precessor Dict:", precessorDict)

    hyperedgeDict = {
        "e1": ["a", "b"],
        "v1": ["a"],
        "v2": ["b"],
        "e2": ["a", "b", "c"]
    }

    clusterHyperedgeDict, clusterVariablesDict, successorDict, precessorDict = generateClusterTree(
        hyperedgeDict, variableEliminationList=["a", "c", "b"], clusterSuf=clusterSuf)
    mesSched = find_message_schedule_towards_root(successorDict, precessorDict)

    assert mesSched[0] == ('a' + clusterSuf, 'c' + clusterSuf)
    assert mesSched[1] == ('c' + clusterSuf, 'b' + clusterSuf)
    assert len(mesSched) == 2