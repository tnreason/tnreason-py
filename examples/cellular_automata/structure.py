from tnreason import application
import numpy as np
from matplotlib import pyplot as plt

aSuf = "_aVar"

def compute_state(queryVariable, successorActivationDict, inferer):
    return inferer.query(variableList=[queryVariable], evidenceDict=successorActivationDict).draw_sample()

def compute_evolution(startVector, inferer, evolNum=10, neighborInfluence=(2, 2)):
    cellNumber = len(startVector)
    evolStates = np.empty((evolNum, cellNumber))
    evolStates[0] = startVector
    for evolPos in range(1, evolNum):
        for cellPos in range(cellNumber):
            valDict = {**{"p" + str(i): evolStates[evolPos - 1, (cellPos + i) % cellNumber] for i in
                          range(1, neighborInfluence[0])},
                       **{"m" + str(i): evolStates[evolPos - 1, (cellPos - i) % cellNumber] for i in
                          range(1, neighborInfluence[1])},
                       "b": evolStates[evolPos - 1, cellPos]}
            evolStates[evolPos, cellPos] = compute_state("n" + aSuf, valDict, inferer)["n" + aSuf]
    return evolStates

def visualize_evolution(evolStates, storePath=None):
    plt.imshow(evolStates, vmin=0, vmax=1, cmap="binary")
    plt.title("Simulated Cellular Automata")
    plt.xticks([])
    plt.yticks([])
    if storePath is not None:
        plt.savefig(storePath)
    else:
        plt.show()

if __name__ == "__main__":
    inferer = application.InferenceProvider(
        application.HybridKnowledgeBase(
            facts={"f1": ["eq", "m1", "n"]}
        )
    )

    assert compute_state("n" + aSuf, {"oldRight": 1, "m1": 0}, inferer)["n" + aSuf] == 0
    assert compute_state("n" + aSuf, {"oldRight": 1, "m1": 1}, inferer)["n" + aSuf] == 1

    visualize_evolution(compute_evolution([0, 0, 1, 0] + [0 for i in range(10)], inferer, evolNum=100))