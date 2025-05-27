import unittest

from tnreason import engine

from tnreason.engine import polynomial_handling as ph
from tnreason.engine import workload_to_gurobi as ptg


class GurobiTest(unittest.TestCase):
    def test_binarization(self):
        polyCore = engine.get_core("PolynomialCore")(
            values=[(-2, {"c1": 0, "c2": 1}), (3.4, {"c2": 0}), (11.123, {"c1": 1})],
            shape=[3, 2],
            colors=["c1", "c2"])
        binP = ph.binarize_polyCore(polyCore)
        model = ptg.core_to_gurobi_model(binP)
        model.optimize()

        self.assertEqual(model.getVarByName("c2").X, 0)
        self.assertEqual(model.getVarByName("c1_1").X, 1)

    def test_implication(self):
        from tnreason.representation import cnf_to_cores as ctc

        polyCore = ctc.weightedFormulas_to_sparseCore({
            "w1": ["imp", "a", "b", 0.678],
            "w2": ["a", 0.34]
        })

        model = ptg.core_to_gurobi_model(polyCore)
        model.optimize()

        self.assertEqual(model.getVarByName("a").X, 1)
        self.assertEqual(model.getVarByName("b").X, 1)

    def test_argmax(self):
        from tnreason.representation import cnf_to_cores as ctc

        polyCore = ctc.weightedFormulas_to_sparseCore({
            "w1": ["imp", "a", "b", 0.678],
            "w2": ["a", 0.34]
        })

        optimum = polyCore.get_argmax()
        self.assertEqual(optimum["a"], 1)
        self.assertEqual(optimum["b"], 1)
