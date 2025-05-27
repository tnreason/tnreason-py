import unittest

from tnreason.representation import cnf_to_cores as ctc


class CNFTest(unittest.TestCase):
    def test_clause_representation(self):
        for coreType in ["PolynomialCore", "NumpyCore", "PandasCore"]:
            clauseCore = ctc.clause_to_core({"c": 0, "b": 1}, coreType=coreType)
            self.assertEqual(clauseCore[0, 0], 1)
            self.assertEqual(clauseCore[1, 0], 0)
