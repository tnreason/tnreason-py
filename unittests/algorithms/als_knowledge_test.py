import unittest

from tnreason import representation
from tnreason import engine

import numpy as np

from tnreason.reasoning import alternating_least_squares as als

aSuf = representation.suf.disVarSuf

networkCores = {
    **representation.create_raw_formula_cores(["imp", "a1", "a2"])
}

targetCores = {
    **als.copy_cores(representation.create_formulas_cores({"f1": ["imp", "a1", "a2"]}), "_tar", ["a1" + aSuf, "a2" + aSuf]),
}

optimizer = als.ALS(
    networkCores=networkCores,
    targetCores=targetCores,
    importanceColors=["a1" + aSuf, "a2" + aSuf]
)


class AlsKnowledgeTest(unittest.TestCase):
    # def test_operator_check(self):
    #     conOperator = optimizer.compute_conOperator(
    #         updateColors=["(imp_a1_a2)" + representation.suf.comVarSuf])
    #     self.assertEqual(conOperator.values[1, 1], 3)
    #     self.assertEqual(conOperator.values[1, 0], 0)
    #     self.assertEqual(conOperator.values[0, 1], 0)
    #     self.assertEqual(conOperator.values[0, 0], 1)
    #
    #     operator = engine.contract(coreDict={
    #         **networkCores,
    #         **als.copy_cores(networkCores, "_out",
    #                          ["a1" + representation.suf.disVarSuf, "a2" + representation.suf.disVarSuf])},
    #         openColors=["(imp_a1_a2)" + representation.suf.comVarSuf,
    #                     "(imp_a1_a2)" + representation.suf.comVarSuf + "_out"])
    #
    #     self.assertEqual(operator.values[0, 0], conOperator.values[0, 0])
    #     self.assertEqual(operator.values[0, 1], conOperator.values[0, 1])

    def test_world_recovery(self):
        optimizer.random_initialize(["estHead"], {"estHead": 2},
                                    {"estHead": ["(imp_a1_a2)" + representation.suf.comVarSuf]})
        optimizer.alternating_optimization(["estHead"], computeResiduum=False, sweepNum=1)

        self.assertEqual(optimizer.networkCores["estHead"].values[0], 0)
        self.assertEqual(optimizer.networkCores["estHead"].values[1], 1)

    def test_data_recovery(self):
        dataNum = 4
        data = np.zeros(shape=(2, 2, dataNum))
        data[0, 0, 0] = 1
        data[1, 1, 1] = 1
        data[0, 1, 2] = 1
        data[1, 1, 3] = 1

        dataOptimizer = als.ALS(
            networkCores=representation.create_raw_formula_cores(["imp", "a1", "a2"]),
            targetCores={"tarCore": engine.get_core()(values=np.ones(dataNum), colors=["dat"])},
            importanceList=[
                (1, {"dataTensor": engine.get_core()(values=data, colors=["a1" + aSuf, "a2" + aSuf, "dat"])})],
            importanceColors=["a1" + aSuf, "a2" + aSuf]
        )

        dataOptimizer.random_initialize(["estHead"], {"estHead": 2},
                                        {"estHead": ["(imp_a1_a2)" + representation.suf.comVarSuf]})
        dataOptimizer.alternating_optimization(["estHead"], computeResiduum=False, sweepNum=1)
        self.assertEqual(dataOptimizer.networkCores["estHead"].values[0], 0)
        self.assertEqual(dataOptimizer.networkCores["estHead"].values[1], 1)
