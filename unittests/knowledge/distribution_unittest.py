import unittest

from tnreason import engine, knowledge

from tnreason.knowledge import distributions as dist

methodList = [{"coreType": "NumpyCore", "contractionMethod": "NumpyEinsum"},
              {"coreType": "PolynomialCore", "contractionMethod": "CorewiseContractor"},
              {"coreType": "PandasCore", "contractionMethod": "CorewiseContractor"}
              ]


class MNTest(unittest.TestCase):

    def test_single_core(self):
        valueCore = engine.get_core("PolynomialCore")(
            values=[(1, {"a": 0}), (1.24, {"a": 1, "b": 0}), (0.36, {"a": 0, "c": 4})
                    ], shape=[2, 3, 5], colors=["a", "b", "c"])

        for method in methodList:
            mN = dist.MarkovNetwork(coreDict={"core": engine.convert(valueCore, method["coreType"])},
                                    contractionMethod=method["contractionMethod"], coreType=method["coreType"])
            self.assertEqual(mN.get_partition_function(addDimDict={"d": 10}), 222.8)
            self.assertEqual(mN.get_partition_function(), 22.28)

            marginal = engine.contract(mN.create_cores(), openColors=["a", "c"],
                                       contractionMethod=method["contractionMethod"])
            self.assertEqual(marginal[0, 4], 3 * (1 + 0.36))

    def test_empirical(self):
        methodList = [{"coreType": "NumpyCore", "contractionMethod": "NumpyEinsum"},
                      {"coreType": "PolynomialCore", "contractionMethod": "CorewiseContractor"},
                      #{"coreType": "PandasCore", "contractionMethod": "CorewiseContractor"} #Takes too long!
                      ]
        for method in methodList:
            generatingKB = knowledge.InferenceProvider(knowledge.HybridKnowledgeBase(weightedFormulas=
            {
                "f1": ["imp", "a", "b", 2.567],
                "f2": ["imp", "a", "c", 2.222],
                "f3": ["a", 1.78]
            }, coreType=method["coreType"], contractionMethod=method["contractionMethod"]),
                coreType=method["coreType"], contractionMethod=method["contractionMethod"]
            )

            sampleNum = 10
            sampleDf = generatingKB.draw_samples(sampleNum, dfOutput=True)

            knowledge.get_empirical_distribution(sampleDf)


if __name__ == "__main__":
    unittest.main()
