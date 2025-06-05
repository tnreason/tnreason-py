import unittest

import tnreason.representation.coordinate_calculus
from tnreason import representation

from tnreason.reasoning import variational_inference as ib
from tnreason.representation import features as ed

from tnreason.application import  formulas_to_cores as ftc

precisionTolerance = 1e-5

f1 = ["and", "a1", "a2"]
f2 = ["not", "a3"]

computationCores = {
    **ftc.create_formula_computation_cores(f1),
    **ftc.create_formula_computation_cores(f2)}

caNetwork = representation.ComputationActivationNetwork(
    featureDict={
        "(and_a1_a2)_cV": representation.SingleSoftFeature(featureColor="(and_a1_a2)_cV", affectedComputationCores=["(and_a1_a2)_cC"],
                                               name="f1"),
        "(not_a3)_cV": representation.SingleSoftFeature(featureColor="(not_a3)_cV", affectedComputationCores=["(not_a3)_cC"],
                                            name="f2"),
        "a3_dV": representation.SoftPartitionFeature(featureColors=["a3_dV"], affectedComputationCores=[], name="f3")
    },
    computationCoreDict={
        **ftc.create_formula_computation_cores(f1),
        **ftc.create_formula_computation_cores(f2),
    }
)

class ForwardBackwardTest(unittest.TestCase):
    def test_partition_single_overlap(self):
        tbMatched = representation.create_vanishing_core(name="a3_dV" + representation.suf.actCoreSuf, shape=[2], colors=["a3_dV"])
        tbMatched[{"a3_dV": 0}] = 0.1231
        tbMatched[{"a3_dV": 1}] = 1 - tbMatched[{"a3_dV": 0}]

        meanParamDict = {"(and_a1_a2)_cV": 0.91234,
                         "(not_a3)_cV": tbMatched[{"a3_dV": 0}],
                         "a3_dV": tbMatched}
        bContractor = ib.BackwardAlternator(caNetwork=caNetwork,
                                            meanParamDict=meanParamDict)
        bContractor.alternating_updates()

        matchedMeans = bContractor.forwardInferer.compute_environmentMean("a3_dV")
        self.assertGreaterEqual(precisionTolerance, abs(matchedMeans[0] - tbMatched[{"a3_dV": 0}]))
        self.assertGreaterEqual(precisionTolerance, abs(matchedMeans[1] - tbMatched[{"a3_dV": 1}]))

        self.assertGreaterEqual(precisionTolerance,
                                abs(bContractor.forwardInferer.infer_meanParam("(and_a1_a2)_cV") - meanParamDict[
                                    "(and_a1_a2)_cV"]))
