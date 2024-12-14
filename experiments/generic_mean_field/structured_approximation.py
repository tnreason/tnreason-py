from tnreason import knowledge
from tnreason.algorithms import energy_based_algorithms as eba

energyDict = knowledge.HybridKnowledgeBase(
        weightedFormulas={"w1": ["imp", "a", "b", 5.678],
                          #"w2": ["a", 1]
                          }
    ).get_energy_dict()

approximator = eba.GenericMeanFieldApproximator(energyDict, colors=["a_aVar", "b_aVar"], dimensionDict={"a_aVar": 2, "b_aVar": 2},
                                                edgeColorDict={"e1": ["a_aVar", "b_aVar"]})
approximator.update_core("e1")
print(approximator.approxCores["e1"].values)

print(approximator.get_energyDict())