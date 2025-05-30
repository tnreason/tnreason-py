from tnreason import application
from tnreason.application import inductive as ind

from tnreason import reasoning

# testNet = application.HybridKnowledgeBase(
#     weightedFormulas={"w1": ["not", "a3", 2],
#                       "w2": ["a2", -1]}).to_caNetwork()
#
# infer = reasoning.get_inferer(None)(testNet)
#
# print(infer)

generatingKB = application.InferenceProvider(application.HybridKnowledgeBase(weightedFormulas=
{
    "f1": ["imp", "a", "b", 20.567],
    "f2": ["imp", "a", "c", 2.222],
    "f3": ["a", 1.78],
    "f4": ["b", -100]
}))

sampleNum = 10
sampleDf = generatingKB.draw_samples(sampleNum, dfOutput=True)

empDistribution = application.get_empirical_distribution(sampleDf)

satisfactionDict = ind.calculate_satisfactionDict(empDistribution,
                               {"f1": ["imp", "a", "b"],
                                "f2": ["a"]
                                })

assert satisfactionDict["f1"] == 1 and satisfactionDict["f2"] == 0



learner = application.HybridLearner(application.HybridKnowledgeBase(
    weightedFormulas={"w12": ["imp","c","a", 2],
                      "w213": ["a2", -1]}
))

learner.infer_weights_on_data(empDistribution)

matchedWeightedFormulas = learner.knowledgeBase.weightedFormulas

assert matchedWeightedFormulas["w12"][-1] < 0
assert matchedWeightedFormulas["w213"][-1] == 0
