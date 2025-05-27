from tnreason import application, engine

genKB = application.HybridKnowledgeBase(
    facts={"f1": ["a1"]},
    weightedFormulas={
        "wf1": ["imp", "a1", "a2", 1.1424],
        "wf1.5": ["a2", 0.2],
        "wf2": ["not", "a3", 50.2]
    }
)
samples = application.InferenceProvider(genKB).draw_samples(100, dfOutput=False)
posDict = engine.convert(samples, "PandasCore")

learner = application.HybridLearner(application.HybridKnowledgeBase(
    weightedFormulas={"w1": ["not", "a3", 2],
                      "w2": ["a2", -1]}
))

architecture = {"neur1": [["imp"],
                          ["a1"],
                          ["a3", "a2"]]
                }
headNeuron = "neur1"

learner.initialize_proposalDistribution(architecture, headNeuron,
                                        application.MarkovNetwork({"samples": posDict}, coreType="PandasCore",
                                                                  contractionMethod="CorewiseContractor"),
                                        alternationMethod="forwardSampling")
testDict = application.MarkovNetwork({"samples": posDict})
proposalEnergy = learner.proposalDistribution.get_energy_dict()

print(learner.propose_candidate(architecture=architecture, optimizationMethod="exactEnergyMax"))
