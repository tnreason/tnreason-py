from tnreason import reasoning, engine, application
from experiments.variable_elimination import generate_random_mn as gr
from experiments.variable_elimination import generate_clusterTree as gc

def find_message_schedule(successorDict, precessorDict):
    messageTuples = []
    while any([len(successorDict[variableClusterKey])>0 for variableClusterKey in precessorDict]):
        for variableClusterKey in list(successorDict):
            if len(precessorDict[variableClusterKey]) == 0:
                for nextVariable in successorDict[variableClusterKey]:
                    messageTuples.append((variableClusterKey,nextVariable))
                for nextVariable in successorDict[variableClusterKey]:
                    precessorDict[nextVariable].remove(variableClusterKey)
                successorDict[variableClusterKey] = []
    return messageTuples



if __name__ == "__main__":
    hyperedgeDict = {
        "e1": ["a", "b"],
        "v1": ["a"],
        "v2": ["b"],
        "e2": ["a", "b", "c"]
    }
    # randomTN = gr.generate_random_positive_tn(
    #     hyperedgeDict=hyperedgeDict,
    #     shapeDict={"a": 2, "b": 3, "c": 5}
    # )
    # reasoning.get_inferer("ExpectationPropagator")(
    #     caNetwork=application.MarkovNetwork(coreDict=randomTN).create_caNetwork(),
    # )

    clusterHyperedgeDict, clusterVariablesDict, successorDict, precessorDict = gc.generateClusterTree(hyperedgeDict, variableEliminationList=["a", "c", "b"])
    mesSched = find_message_schedule(successorDict, precessorDict)
    assert mesSched[0] == ('a', 'c')
    assert mesSched[1] == ('c', 'b')

    hyperedgeDict = {
        "e1": ["a", "b"],
        "e2": ["a"],
        "e3": ["b"],
        "e4": ["b", "c"],
        "e5": ["c"],
    }
    clusterHyperedgeDict, clusterVariablesDict, successorDict, precessorDict = gc.generateClusterTree(hyperedgeDict, variableEliminationList=["a", "c", "b"])
    mesSched = find_message_schedule(successorDict, precessorDict)
    assert mesSched[0] == ('a', 'b')
    assert mesSched[1] == ('c', 'b')