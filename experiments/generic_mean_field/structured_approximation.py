from tnreason import application
from tnreason.reasoning import energy_based_algorithms as eba

energyDict = application.HybridKnowledgeBase(
    weightedFormulas={"w1": ["imp", "a", "b", 5.678],
                      # "w2": ["a", 1]
                      }
).get_energy_dict()

#from tnreason import engine
#energyDict = {"w": [{"core": engine.get_core("PolynomialCore")(values=[(1, {"a_aVar": 1})], colors=["a_aVar"], shape=[2])}, 0.56]}

approximator = eba.GenericMeanFieldApproximator(energyDict, colors=["a_aVar", "b_aVar"],
                                                dimensionDict={"a_aVar": 2, "b_aVar": 2},
                                                edgeColorDict={"e1": ["a_aVar", "b_aVar"]},
                                               # coreType="PolynomialCore",
                                               # contractionMethod="CorewiseContractor",
                                                dimDict={"a_aVar" : 2, "b_aVar" : 2})
approximator.update_core("e1")
print(approximator.approxCores["e1"].values)

#print(approximator.get_energyDict())
#eba.EnergyGibbsSampleCore(approximator.get_energyDict())
