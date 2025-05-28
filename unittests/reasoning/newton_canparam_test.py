import unittest

from tnreason import representation
from tnreason.reasoning import inference as ib

import math

f1 = ["and", "a1", "a2"]
f2 = ["not", "a3"]

longfeatureLength = 10

computationCores = {
    **representation.create_raw_formula_cores(f1),
    **representation.create_raw_formula_cores(f2),
    # "tCore" : representation.create_trivial_core("longFeature" + representation.suf.actCoreSuf, [longfeatureLength], ["longFeature"])
}

# engine.draw_factor_graph(computationCores)

singleFeatureDict = {"(and_a1_a2)_cV": ["(and_a1_a2)_cC"],
                     "(not_a3)_cV": ["(not_a3)_cC"],
                     "longFeature": []}
partitionFeatureDict = {"a3_dV": []}

interpretationDict = {
    "a3_dV": [0, 1],
    "(not_a3)_cV": [0, 1],
    "longFeature": [i for i in range(longfeatureLength)],
    "(and_a1_a2)_cV": [0, 1],
}

fContractor = ib.ForwardContractor(computationCores=computationCores,
                                   singleFeatureDict=singleFeatureDict,
                                   partitionFeatureDict=partitionFeatureDict,
                                   interpretationDict=interpretationDict,
                                   dimensionDict={"(and_a1_a2)_cV": 2, "a3_dV": 2, "longFeature": longfeatureLength})

## Not in all_unittests, auxiliary test
class NewtonTest(unittest.TestCase):
    def test_uncomputed_longerfeature(self):
        tbMatched = 0 * representation.create_trivial_core("a3_dV" + representation.suf.actCoreSuf, [2], ["a3_dV"])
        tbMatched[{"a3_dV": 0}] = 0.1231
        tbMatched[{"a3_dV": 1}] = 1 - tbMatched[{"a3_dV": 0}]

        longMean = 4.5
        meanParamDict = {"(and_a1_a2)_cV": 0.91234,
                         "(not_a3)_cV": tbMatched[{"a3_dV": 0}],
                         "a3_dV": tbMatched,
                         "longFeature": longMean}

        bContractor = ib.BackwardAlternator(computationCores=computationCores, singleFeatureDict=singleFeatureDict,
                                            partitionFeatureDict=partitionFeatureDict,
                                            interpretationDict=interpretationDict,
                                            forwardInferer=fContractor, canparamDict=None, meanparamDict=meanParamDict,
                                            dimensionDict={"(and_a1_a2)_cV": 2, "a3_dV": 2,
                                                           "longFeature": longfeatureLength})
        bContractor.update_single_feature("longFeature")
        self.assertTrue(bContractor.canParamDict["longFeature"] == 0)

        longMean = 5.75 # Stability problems for larger realizable means, tune the dumpFactor?
        meanParamDict = {"(and_a1_a2)_cV": 0.91234,
                         "(not_a3)_cV": tbMatched[{"a3_dV": 0}],
                         "a3_dV": tbMatched,
                         "longFeature": longMean}

        bContractor = ib.BackwardAlternator(computationCores=computationCores, singleFeatureDict=singleFeatureDict,
                                            partitionFeatureDict=partitionFeatureDict,
                                            interpretationDict=interpretationDict,
                                            forwardInferer=fContractor, canparamDict=None, meanparamDict=meanParamDict,
                                            dimensionDict={"(and_a1_a2)_cV": 2, "a3_dV": 2,
                                                           "longFeature": longfeatureLength})
        bContractor.update_single_feature("longFeature")
        self.assertTrue(bContractor.canParamDict["longFeature"] > 0)

        longMean = 2.15
        meanParamDict = {"(and_a1_a2)_cV": 0.91234,
                         "(not_a3)_cV": tbMatched[{"a3_dV": 0}],
                         "a3_dV": tbMatched,
                         "longFeature": longMean}

        bContractor = ib.BackwardAlternator(computationCores=computationCores, singleFeatureDict=singleFeatureDict,
                                            partitionFeatureDict=partitionFeatureDict,
                                            interpretationDict=interpretationDict,
                                            forwardInferer=fContractor, canparamDict=None, meanparamDict=meanParamDict,
                                            dimensionDict={"(and_a1_a2)_cV": 2, "a3_dV": 2,
                                                           "longFeature": longfeatureLength})
        bContractor.update_single_feature("longFeature")
        self.assertTrue(bContractor.canParamDict["longFeature"] < 0)
