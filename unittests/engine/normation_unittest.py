import unittest

from tnreason import engine

from tnreason import representation

methodList = [{"coreType": "NumpyCore", "contractionMethod": "NumpyEinsum"},
              #{"coreType": "PolynomialCore", "contractionMethod": "CorewiseContractor"}, # Both Polynomial and Pandas Cores do not handle normation of trivial core (1, dict()) correctly!
              #{"coreType": "PandasCore", "contractionMethod": "CorewiseContractor"}
              ]

aSuf = representation.suf.disVarSuf

from tnreason.application import formulas_to_cores as ftc

class NormationTest(unittest.TestCase):
    def test_hard_head(self):
        for method in methodList:
            cores = ftc.create_formulas_cores({"f1": ["and", "a", "b", 1.23426]},
                                                         coreType=method["coreType"])
            normed = engine.normate(cores, outColors=["(and_a_b)" + representation.suf.comVarSuf],
                                    inColors=["a" + aSuf, "b" + aSuf], contractionMethod=method["contractionMethod"])
            self.assertEqual(normed[0, 0, 0], 1)
            self.assertEqual(normed[1, 0, 0], 0)

    def test_empty_normation(self):
        for method in methodList:
            normed = engine.normate({}, outColors=["madeUp1", "madeUp2"], inColors=["madeUp3"],
                                    contractionMethod=method["contractionMethod"],
                                    coreType=method["coreType"])
            self.assertEqual(normed[0, 0, 0], 0.25)
            self.assertEqual(normed[1, 0, 1], 0.25)

    def test_empty_contraction(self):
        methodList = [{"coreType": "NumpyCore", "contractionMethod": "NumpyEinsum"},
                      {"coreType": "PolynomialCore", "contractionMethod": "CorewiseContractor"},
                      {"coreType": "PandasCore", "contractionMethod": "CorewiseContractor"}
                      ]
        for method in methodList:
            trivialCore = engine.create_from_slice_iterator(shape=[2, 3, 5], colors=["piskle", "jezyk", "bazant"],
                                                            sliceIterator=[(1, dict())],
                                                            coreType=method["coreType"])

            self.assertEqual(
                engine.contract({"triv": trivialCore}, openColors=[], contractionMethod=method["contractionMethod"])[:],
                30)

            trivialCore = engine.create_from_slice_iterator(shape=[2, 2], colors=["madeUp1", "madeUp2"],
                                                            sliceIterator=[(1, dict())],
                                                            coreType=method["coreType"])
            self.assertEqual(engine.contract({}, openColors=["madeUp1", "madeUp2"],
                                    contractionMethod=method["contractionMethod"],
                                    coreType=method["coreType"]), trivialCore)