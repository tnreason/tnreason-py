def generateClusterTree(hyperedgeDict, variableEliminationList):
    clusterHyperedgeDict = {variableKey : [] for variableKey in variableEliminationList}
    clusterVariablesDict = {variableKey : set() for variableKey in variableEliminationList}

    assignedHyperEdges = []
    openClusters   = []

    successorDict = {variableKey : [] for variableKey in variableEliminationList}
    precessorDict = {variableKey : [] for variableKey in variableEliminationList}

    for variableKey in variableEliminationList:

        for hyperKey in hyperedgeDict:
            if hyperKey not in assignedHyperEdges:
                if variableKey in hyperedgeDict[hyperKey]:
                    clusterHyperedgeDict[variableKey].append(hyperKey)
                    clusterVariablesDict[variableKey] = clusterVariablesDict[variableKey] | set(hyperedgeDict[hyperKey])
                    assignedHyperEdges.append(hyperKey)

        for potentialPre in openClusters:
            if variableKey in clusterVariablesDict[potentialPre]:
                successorDict[potentialPre].append(variableKey)
                precessorDict[variableKey].append(potentialPre)

                clusterVariablesDict[variableKey] = clusterVariablesDict[variableKey] | clusterVariablesDict[potentialPre]
                clusterVariablesDict[variableKey].remove(potentialPre)  # Remove predecessor from variable cluster

        for pre in precessorDict[variableKey]:
            openClusters.remove(pre) # To not jump over a potentialPre when deleting while iterating in the above loop

        openClusters.append(variableKey)

    return clusterHyperedgeDict, clusterVariablesDict, successorDict, precessorDict

if __name__ == "__main__":
    hyperedgeDict = {
        "e1": ["a", "b"],
        "e2": ["a"],
        "e3": ["b"],
        "e4": ["b", "c"],
        "e5": ["c"],
    }
    variableEliminationList = ["a", "c", "b"]

    clusterHyperedgeDict, clusterVariablesDict, successorDict, precessorDict = generateClusterTree(hyperedgeDict, variableEliminationList)

    print("Cluster Hyperedge Dict:", clusterHyperedgeDict)
    print("Cluster Variables Dict:", clusterVariablesDict)
    print("Successor Dict:", successorDict)
    print("Precessor Dict:", precessorDict)