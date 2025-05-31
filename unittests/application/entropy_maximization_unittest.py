import unittest

from tnreason import application

import numpy as np

generatingKB = application.InferenceProvider(application.HybridKnowledgeBase(weightedFormulas=
{
    "f1": ["imp", "a", "b", 2.567],
    "f2": ["imp", "a", "c", 2.222],
    "f3": ["a", 1.78]
}))

sampleNum = 200
sampleDf = generatingKB.draw_samples(sampleNum, dfOutput=True)

class WeightEstimationTest(unittest.TestCase):
    def test_convergence(self):
        expressionsDict = {"f1": ["imp", "a", "b"], "f2": ["imp", "a", "c"], "f3": ["a"]}
        hybridKB = application.HybridKnowledgeBase(
            weightedFormulas={key: expressionsDict[key] + [0] for key in expressionsDict})
        calibrator = application.HybridLearner(hybridKB)
        weights = calibrator.infer_weights_on_data(application.get_empirical_distribution(sampleDf), satInferenceMethod="ForwardContractor")

        for key in weights:
            self.assertGreaterEqual(0.1, abs(weights[key][-1] - weights[key][-2]))

    def test_infinity_handling(self):
        expressionsDict = {"f1": ["imp", "a", "b"], "f2": ["imp", "a", "c"], "f3": ["a"]}

        hybridKB = application.HybridKnowledgeBase(
            weightedFormulas={key: expressionsDict[key] + [0] for key in expressionsDict},
            facts={"fact1": ["imp", "a", "b"]})
        calibrator = application.HybridLearner(hybridKB)
        weights = calibrator.infer_weights_on_data(application.get_empirical_distribution(sampleDf), satInferenceMethod="ForwardContractor")

        self.assertEqual(0, weights["f1"][0]) # Since feature cannot be tuned, check whether it is not changed
        self.assertEqual(0, weights["f1"][1])

    def test_atomic_variables(self):
        expressionsDict = {"f_a": ["a"],
                           "f_b": ["b"],
                           "f_c": ["c"]}
        hybridKB = application.HybridKnowledgeBase(
            weightedFormulas={key: expressionsDict[key] + [0] for key in expressionsDict})
        calibrator = application.HybridLearner(hybridKB)
        weights = calibrator.infer_weights_on_data(application.get_empirical_distribution(sampleDf), satInferenceMethod="ForwardContractor")

        inferer = application.InferenceProvider(application.get_empirical_distribution(sampleDf))
        satisfactionDict = {key: inferer.ask(expressionsDict[key]) for key in expressionsDict}

        self.assertAlmostEquals(satisfactionDict["f_a"],
                                np.exp(weights["f_a"][1]) / (1 + np.exp(weights["f_a"][1])),
                                delta=0.01)
        self.assertAlmostEquals(satisfactionDict["f_b"],
                                np.exp(weights["f_b"][1]) / (1 + np.exp(weights["f_b"][1])),
                                delta=0.01)
        self.assertAlmostEquals(satisfactionDict["f_c"],
                                np.exp(weights["f_c"][1]) / (1 + np.exp(weights["f_c"][1])),
                                delta=0.01)

    def test_kb_with_data_backCores(self):
        expressionsDict = {"f_a": ["a"],
                           "f_b": ["b"],
                           "f_c": ["c"]}

        hybridKB = application.HybridKnowledgeBase(
            weightedFormulas={key: expressionsDict[key] + [0] for key in expressionsDict})
        calibrator = application.HybridLearner(hybridKB)
        weights = calibrator.infer_weights_on_data(application.get_empirical_distribution(sampleDf), satInferenceMethod="ForwardContractor")

        if not "f_a" in weights:
            self.assertTrue("f_a" in calibrator.knowledgeBase.facts)
        if not "f_b" in weights:
            self.assertTrue("f_b" in calibrator.knowledgeBase.facts)
        if not "f_c"  in weights:
            self.assertTrue("f_c" in calibrator.knowledgeBase.facts)