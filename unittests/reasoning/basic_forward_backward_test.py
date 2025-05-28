import unittest

from tnreason import representation
from tnreason.reasoning import inference as ib

f1 = ["and", "a1", "a2"]
f2 = ["not", "a3"]

computationCores = {
    **representation.create_raw_formula_cores(f1),
    **representation.create_raw_formula_cores(f2)}

# engine.draw_factor_graph(computationCores)

singleFeatureDict = {"(and_a1_a2)_cV": ["(and_a1_a2)_cC"],
                     "(not_a3)_cV": ["(not_a3)_cC"]}
partitionFeatureDict = {"a3_dV": []}

fContractor = ib.ForwardContractor(computationCores=computationCores,
                                   singleFeatureDict=singleFeatureDict,
                                   partitionFeatureDict=partitionFeatureDict,
                                   dimensionDict={"(and_a1_a2)_cV": 2, "a3_dV": 2})


class ForwardBackwardTest(unittest.TestCase):
    def test_partition_single_overlap(self):
        tbMatched = 0 * representation.create_trivial_core("a3_dV" + representation.suf.actCoreSuf, [2], ["a3_dV"])
        tbMatched[{"a3_dV": 0}] = 0.1231
        tbMatched[{"a3_dV": 1}] = 1 - tbMatched[{"a3_dV": 0}]

        meanParamDict = {"(and_a1_a2)_cV": 0.91234,
                         "(not_a3)_cV": tbMatched[{"a3_dV": 0}],
                         "a3_dV": tbMatched}
        bContractor = ib.BackwardAlternator(computationCores=computationCores, singleFeatureDict=singleFeatureDict,
                                            partitionFeatureDict=partitionFeatureDict,
                                            forwardInferer=fContractor, canparamDict=None, meanparamDict=meanParamDict,
                                            interpretationDict=None, dimensionDict={"(and_a1_a2)_cV": 2, "a3_dV": 2})
        bContractor.alternating_updates()

        matchedMeans = bContractor.forwardInferer.calculate_indicator_mean("a3_dV")

        self.assertGreaterEqual(1e-5, abs(matchedMeans[0] - tbMatched[{"a3_dV": 0}]))
        self.assertGreaterEqual(1e-5, abs(matchedMeans[1] - tbMatched[{"a3_dV": 1}]))
        self.assertGreaterEqual(1e-5,
                                abs(bContractor.forwardInferer.calculate_single_mean("(and_a1_a2)_cV") - meanParamDict[
                                    "(and_a1_a2)_cV"]))