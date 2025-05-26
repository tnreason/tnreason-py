import unittest

from tnreason import encoding
from tnreason import knowledge

import numpy as np

generatingKB = knowledge.InferenceProvider(knowledge.HybridKnowledgeBase(weightedFormulas=
{
    "f1": ["imp", "a", "b", 2.567],
    "f2": ["imp", "a", "c", 2.222],
    "f3": ["a", 1.78]
}))

sampleNum = 200
sampleDf = generatingKB.draw_samples(sampleNum, dfOutput=True)

hSuf = encoding.suf.actCoreSuf


class WeightEstimationTest(unittest.TestCase):
    def test_convergence(self):
        expressionsDict = {"f1": ["imp", "a", "b"], "f2": ["imp", "a", "c"], "f3": ["a"]}
        hybridKB = knowledge.HybridKnowledgeBase(
            weightedFormulas={key: expressionsDict[key] + [0] for key in expressionsDict})
        calibrator = knowledge.WeightEstimator(hybridKB)
        calibrator.get_satisfaction_dict(knowledge.get_empirical_distribution(sampleDf))
        calibrator.fact_check()
        weights = calibrator.calibrate_weights(sweepNum=10)
        for key in weights:
            self.assertGreaterEqual(0.1, abs(weights[key][-1] - weights[key][-2]))

        # inferer = knowledge.InferenceProvider(knowledge.get_empirical_distribution(sampleDf))
        # satisfactionDict = {key: inferer.ask(expressionsDict[key]) for key in expressionsDict}
        #
        # entropyMaximizer = knowledge.EntropyMaximizer(expressionsDict, satisfactionDict=satisfactionDict, backCores={})
        # weights, facts = entropyMaximizer.alternating_optimization(sweepNum=10)
        # for key in weights:
        #     self.assertGreaterEqual(0.1, abs(weights[key][-1] - weights[key][-2]))

    def test_infinity_handling(self):
        expressionsDict = {"f1": ["imp", "a", "b"], "f2": ["imp", "a", "c"], "f3": ["a"]}

        hybridKB = knowledge.HybridKnowledgeBase(
            weightedFormulas={key: expressionsDict[key] + [0] for key in expressionsDict},
            facts={"fact1": ["imp", "a", "b"]})
        calibrator = knowledge.WeightEstimator(hybridKB)
        calibrator.get_satisfaction_dict(knowledge.get_empirical_distribution(sampleDf))
        calibrator.fact_check()
        weights = calibrator.calibrate_weights(sweepNum=10)

        self.assertEqual(0, weights["f1" + hSuf][0])
        self.assertEqual(0, weights["f1" + hSuf][1])

        # inferer = knowledge.InferenceProvider(knowledge.get_empirical_distribution(sampleDf))
        # satisfactionDict = {key: inferer.ask(expressionsDict[key]) for key in expressionsDict}
        #
        # preBackCores = encoding.create_formulas_cores({"fact1": ["imp", "a", "b"]})
        # backCores = {key + "_back": preBackCores[key] for key in preBackCores}
        #
        # entropyMaximizer = knowledge.EntropyMaximizer(expressionsDict, satisfactionDict=satisfactionDict,
        #                                               backCores=backCores)
        # weights, facts = entropyMaximizer.alternating_optimization(sweepNum=2)
        #
        # self.assertEqual(0, weights["f1"][0])
        # self.assertEqual(0, weights["f1"][1])

    def test_atomic_variables(self):
        expressionsDict = {"f_a": ["a"],
                           "f_b": ["b"],
                           "f_c": ["c"]}
        hybridKB = knowledge.HybridKnowledgeBase(
            weightedFormulas={key: expressionsDict[key] + [0] for key in expressionsDict})
        calibrator = knowledge.WeightEstimator(hybridKB)
        calibrator.get_satisfaction_dict(knowledge.get_empirical_distribution(sampleDf))
        calibrator.fact_check()
        weights = calibrator.calibrate_weights(sweepNum=2)

        inferer = knowledge.InferenceProvider(knowledge.get_empirical_distribution(sampleDf))
        satisfactionDict = {key: inferer.ask(expressionsDict[key]) for key in expressionsDict}

        self.assertAlmostEquals(satisfactionDict["f_a"],
                                np.exp(weights["f_a" + hSuf][1]) / (1 + np.exp(weights["f_a" + hSuf][1])),
                                delta=0.01)
        self.assertAlmostEquals(satisfactionDict["f_b"],
                                np.exp(weights["f_b" + hSuf][1]) / (1 + np.exp(weights["f_b" + hSuf][1])),
                                delta=0.01)
        self.assertAlmostEquals(satisfactionDict["f_c"],
                                np.exp(weights["f_c" + hSuf][1]) / (1 + np.exp(weights["f_c" + hSuf][1])),
                                delta=0.01)

        # entropyMaximizer = knowledge.EntropyMaximizer(expressionsDict, satisfactionDict=satisfactionDict)
        # weights, facts = entropyMaximizer.alternating_optimization(sweepNum=2)
        #
        #
        # self.assertAlmostEquals(satisfactionDict["f_a"], np.exp(weights["f_a"][1]) / (1 + np.exp(weights["f_a"][1])),
        #                         delta=0.01)
        # self.assertAlmostEquals(satisfactionDict["f_b"], np.exp(weights["f_b"][1]) / (1 + np.exp(weights["f_b"][1])),
        #                         delta=0.01)
        # self.assertAlmostEquals(satisfactionDict["f_c"], np.exp(weights["f_c"][1]) / (1 + np.exp(weights["f_c"][1])),
        #                         delta=0.01)

    def test_kb_with_data_backCores(self):
        expressionsDict = {"f_a": ["a"],
                           "f_b": ["b"],
                           "f_c": ["c"]}

        hybridKB = knowledge.HybridKnowledgeBase(
            weightedFormulas={key: expressionsDict[key] + [0] for key in expressionsDict},
            backCores=knowledge.get_empirical_distribution(sampleDf).create_cores())
        calibrator = knowledge.WeightEstimator(hybridKB)
        calibrator.get_satisfaction_dict(knowledge.get_empirical_distribution(sampleDf))
        calibrator.fact_check()
        weights = calibrator.calibrate_weights(sweepNum=2)

        if "f_a" + hSuf in weights:
            self.assertAlmostEqual(weights["f_a" + hSuf][1], 0, delta=0.000001)
        else:
            self.assertTrue("f_a" in calibrator.hybridKB.facts)
        if "f_b" + hSuf in weights:
            self.assertAlmostEqual(weights["f_b" + hSuf][1], 0, delta=0.000001)
        else:
            self.assertTrue("f_b" in calibrator.hybridKB.facts)
        if "f_c" + hSuf in weights:
            self.assertAlmostEqual(weights["f_c" + hSuf][1], 0, delta=0.000001)
        else:
            self.assertTrue("f_c" in calibrator.hybridKB.facts)

        # hybridKB = knowledge.HybridKnowledgeBase(backCores=knowledge.get_empirical_distribution(sampleDf).create_cores())
        #
        # inferer = knowledge.InferenceProvider(hybridKB)
        # satisfactionDict = {key: inferer.ask(expressionsDict[key]) for key in expressionsDict}
        #
        # entropyMaximizer = knowledge.EntropyMaximizer(expressionsDict, satisfactionDict=satisfactionDict,
        #                                               backCores=hybridKB.create_hard_cores())
        # weights, facts = entropyMaximizer.alternating_optimization(sweepNum=2)
        #
        # if "f_a" in weights:
        #     self.assertAlmostEqual(weights["f_a"][1], 0, delta=0.000001)
        # else:
        #     self.assertTrue("f_a" in facts)
        # if "f_b" in weights:
        #     self.assertAlmostEqual(weights["f_b"][1], 0, delta=0.000001)
        # else:
        #     self.assertTrue("f_b" in facts)
        # if "f_c" in weights:
        #     self.assertAlmostEqual(weights["f_c"][1], 0, delta=0.000001)
        # else:
        #     self.assertTrue("f_c" in facts)
