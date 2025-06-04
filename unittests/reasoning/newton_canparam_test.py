import unittest

from tnreason import representation
from tnreason.reasoning import variational_inference as ib

from tnreason.representation import features as ed

f1 = ["and", "a1", "a2"]
f2 = ["not", "a3"]

longfeatureLength = 10

computationCores = {
    **representation.create_formula_computation_cores(f1),
    **representation.create_formula_computation_cores(f2)
}

featureDict = {
    "(and_a1_a2)_cV": ed.SingleSoftFeature(featureColor="(and_a1_a2)_cV", affectedComputationCores=["(and_a1_a2)_cC"],
                                           name="f1"),
    "(not_a3)_cV": ed.SingleSoftFeature(featureColor="(not_a3)_cV", affectedComputationCores=["(not_a3)_cC"],
                                        name="f2"),
    "longFeature": ed.SingleSoftFeature(featureColor="longFeature", affectedComputationCores=[],
                                        name="f3",
                                        interpretedImage=[i for i in range(longfeatureLength)],
                                        interpretationDict={"longFeature": [i for i in range(longfeatureLength)]}
                                        ),
    "a3_dV": ed.SoftPartitionFeature(featureColors=["a3_dV"], affectedComputationCores=[], name="f3")
}

dimensionDict = {
    "(and_a1_a2)_cV": 2, "a3_dV": 2, "longFeature": longfeatureLength
}

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

tbMatched = 0 * representation.create_trivial_core("a3_dV" + representation.suf.actCoreSuf, [2], ["a3_dV"])
tbMatched[{"a3_dV": 0}] = 0.1231
tbMatched[{"a3_dV": 1}] = 1 - tbMatched[{"a3_dV": 0}]

meanParamDict = {"(and_a1_a2)_cV": 0.91234,
                 "(not_a3)_cV": tbMatched[{"a3_dV": 0}],
                 "a3_dV": tbMatched,
                 "longFeature": None}

distribution = ed.ComputationActivationNetwork(
    featureDict=featureDict,
    computationCoreDict=computationCores
)

## Not in all_unittests, auxiliary test
class NewtonTest(unittest.TestCase):
    def test_neutral_longerfeature(self):
        meanParamDict["longFeature"] = 4.5

        distribution.canParamDict["longFeature"] = 0

        bContractor = ib.BackwardAlternator(
            caNetwork=distribution,
            forwardInferer=ib.ForwardContractor(caNetwork=distribution, dimensionDict=dimensionDict),
            meanparamDict=meanParamDict,dimensionDict=dimensionDict)

        bContractor.update_canParam("longFeature")
        self.assertTrue(abs(bContractor.caNetwork.canParamDict["longFeature"]) < 1e-5)

    def test_larger_longerfeature(self):
        meanParamDict["longFeature"] = 5.75

        bContractor = ib.BackwardAlternator(
            caNetwork=distribution,
            forwardInferer=ib.ForwardContractor(caNetwork=distribution, dimensionDict=dimensionDict),
            meanparamDict=meanParamDict,dimensionDict=dimensionDict)

        bContractor.update_canParam("longFeature")
        self.assertTrue(bContractor.caNetwork.canParamDict["longFeature"] > 0)

    def test_smaller_longerfeature(self):
        meanParamDict["longFeature"] = 2.15

        bContractor = ib.BackwardAlternator(
            caNetwork=distribution,
            forwardInferer=ib.ForwardContractor(caNetwork=distribution, dimensionDict=dimensionDict),
            meanparamDict=meanParamDict,dimensionDict=dimensionDict)

        bContractor.update_canParam("longFeature")
        self.assertTrue(bContractor.caNetwork.canParamDict["longFeature"] < 0)

