import unittest

from tnreason import application
from tnreason import representation

import pandas as pd

assetsBasePath = "/Users/alexgoessmann/Documents/ENEXA/tnreason/version1/unittests/knowledge/assets/"

backKb = application.load_kb_from_yaml(assetsBasePath + "fb_backKb.yaml")
architecture = representation.load_from_yaml(assetsBasePath + "fb_architecture.yaml")

genKB = application.HybridKnowledgeBase(
    facts={"f1": ["a1"]},
    weightedFormulas={
        "wf1": ["imp", "a1", "a2", 1.1424],
        "wf1.5": ["a2", 0.2],
        "wf2": ["not", "a3", 5.2]
    }
)
sampleDf = application.InferenceProvider(genKB).draw_samples(100, dfOutput=True)


class FormulaBoostingTest(unittest.TestCase):
    def test_yaml_als(self):
        booster = application.Grafter(knowledgeBase=backKb,
                                      specDict={**representation.load_from_yaml(assetsBasePath + "fb_energyMax_boostSpec.yaml"),
                                              "headNeurons": ["neur1"], "architecture": architecture})
        booster.find_candidate(application.get_empirical_distribution(pd.read_csv(assetsBasePath + "fb_sampleDf.csv")))

    def test_yaml_gibbs(self):
        booster = application.Grafter(knowledgeBase=application.load_kb_from_yaml(assetsBasePath + "fb_backKb.yaml"),
                                      specDict={**representation.load_from_yaml(assetsBasePath + "fb_gibbs_boostSpec.yaml"),
                                              "headNeurons": ["neur1"], "architecture": architecture})
        booster.find_candidate(application.get_empirical_distribution(pd.read_csv(assetsBasePath + "fb_sampleDf.csv")))

    def test_exact_implication_finding(self):
        booster = application.Grafter(knowledgeBase=application.load_kb_from_yaml(assetsBasePath + "fb_backKb.yaml"),
                                      specDict={
                                        "method": "exactEnergyMax",
                                        "sweeps": 10,
                                        "headNeurons": ["neur1"],
                                        "architecture":
                                            {"neur1": [["imp"],
                                                       ["a1"],
                                                       ["a3", "a2"]]
                                             },
                                        "acceptanceCriterion": "always",
                                        "calibrationSweeps": 2
                                    })
        booster.find_candidate(application.get_empirical_distribution(sampleDf))
        self.assertEquals(booster.candidates["neur1"][-1], "a2")

    def test_gibbs_implication_finding(self):
        booster = application.Grafter(
            knowledgeBase=application.HybridKnowledgeBase(partitionFunction=2 ** len(backKb.distributedVariables)),
            specDict={
                "method": "gibbsSample",
                "sweeps": 10,
                "headNeurons": ["neur1"],
                "architecture":
                    {"neur1": [["imp"],
                               ["a1"],
                               ["a3", "a2"]]
                     },
                "acceptanceCriterion": "always",
                "calibrationSweeps": 2
            })
        booster.find_candidate(application.get_empirical_distribution(sampleDf))
        print(booster.candidates)
        self.assertEquals(booster.candidates["neur1"][-1], "a2")
