import unittest

from tnreason import engine

from tnreason import encoding

methodList = [{"coreType": "NumpyCore", "contractionMethod": "NumpyEinsum"},
              {"coreType": "PolynomialCore", "contractionMethod": "CorewiseContractor"},
              {"coreType": "PandasCore", "contractionMethod": "CorewiseContractor"}
              ]

aSuf = encoding.suf.atomicVariableSuffix


class NormationTest(unittest.TestCase):
    def test_hard_head(self):
        for method in methodList:
            cores = encoding.create_formulas_cores({"f1": ["and", "a", "b", 1.23426]}, coreType=method["coreType"])
            normed = engine.normate(cores, outColors=["(and_a_b)" + encoding.suf.categoricalVariableSuffix],
                                    inColors=["a" + aSuf, "b" + aSuf], method=method["contractionMethod"])
            self.assertEqual(normed[0, 0, 0], 1)
            self.assertEqual(normed[1, 0, 0], 0)

    def test_empty_normation(self):
        for method in methodList:
            normed = engine.normate({}, outColors=["madeUp1", "madeUp2"], inColors=["madeUp3"],
                                    method=method["contractionMethod"],
                                    coreType=method["coreType"])
            self.assertEqual(normed[0, 0, 0], 0.25)
            self.assertEqual(normed[1, 0, 1], 0.25)
