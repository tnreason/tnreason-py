"""Unit tests for tnreason.solvers and related engine additions.

Covers:
  - core_comparison  : cores_equal, cores_close
  - NumpyEinsumContractor : optimize flag
  - VariableEliminationContractor
  - ForwardChaining  : consistency, propagation, disentanglement
  - backtrack / backtrack_with_restarts : correctness on Latin squares
  - extract_solution : value recovery
"""

import unittest
import numpy as np

from tnreason import engine
from tnreason.engine.workload_to_numpy import NumpyCore, NumpyEinsumContractor
from tnreason.engine.core_comparison import cores_equal, cores_close
from tnreason.solvers import (
    ForwardChaining,
    backtrack,
    backtrack_with_restarts,
    extract_solution,
)

# ── Shared helpers ─────────────────────────────────────────────────────────────

D = 3                                                   # domain size for tests
NEQ3 = (1.0 - np.eye(D)).astype(float)                  # 3×3 NEQ gadget
NEQ4 = (1.0 - np.eye(4)).astype(float)                  # 4×4 NEQ gadget


def _neq(colors, n=D):
    mat = (1.0 - np.eye(n)).astype(float)
    return NumpyCore(mat, list(colors), "neq_" + "_".join(colors))


def _unit(color, val, n=D):
    v = np.zeros(n); v[val] = 1.0
    return NumpyCore(v, [color], f"unit_{color}_{val}")


def _latin3(givens=None):
    """3×3 Latin square TN.  Variables c{i}{j}, domain 0..2."""
    cores = {}
    for i in range(3):
        for j1 in range(3):
            for j2 in range(j1 + 1, 3):
                k = f"row{i}_{j1}{j2}"
                cores[k] = _neq([f"c{i}{j1}", f"c{i}{j2}"])
    for j in range(3):
        for i1 in range(3):
            for i2 in range(i1 + 1, 3):
                k = f"col{j}_{i1}{i2}"
                cores[k] = _neq([f"c{i1}{j}", f"c{i2}{j}"])
    if givens:
        for (r, c), v in givens.items():
            cores[f"giv_{r}{c}"] = _unit(f"c{r}{c}", v)
    return cores


def _is_valid_latin(sol, size=3):
    """Check that sol is a valid size×size Latin square (0-indexed)."""
    for i in range(size):
        if len(set(sol.get(f"c{i}{j}", -1) for j in range(size))) != size:
            return False
    for j in range(size):
        if len(set(sol.get(f"c{i}{j}", -1) for i in range(size))) != size:
            return False
    return True


# ── Test: core_comparison ──────────────────────────────────────────────────────

class TestCoreComparison(unittest.TestCase):

    def _make(self, vals, colors):
        return NumpyCore(np.array(vals, dtype=float), colors, "test")

    def test_cores_equal_same(self):
        c = self._make([[1, 0], [0, 1]], ["a", "b"])
        self.assertTrue(cores_equal(c, c.clone()))

    def test_cores_equal_reordered(self):
        c1 = self._make([[1, 2], [3, 4]], ["a", "b"])
        c2 = self._make([[1, 3], [2, 4]], ["b", "a"])   # transpose
        self.assertTrue(cores_equal(c1, c2))

    def test_cores_equal_different(self):
        c1 = self._make([[1, 0], [0, 1]], ["a", "b"])
        c2 = self._make([[1, 1], [0, 1]], ["a", "b"])
        self.assertFalse(cores_equal(c1, c2))

    def test_cores_close_within_tol(self):
        c1 = self._make([1.0, 2.0, 3.0], ["x"])
        c2 = self._make([1.0 + 1e-9, 2.0, 3.0], ["x"])
        self.assertTrue(cores_close(c1, c2))

    def test_cores_close_outside_tol(self):
        c1 = self._make([1.0, 2.0], ["x"])
        c2 = self._make([1.1, 2.0], ["x"])
        self.assertFalse(cores_close(c1, c2))

    def test_cores_close_different_colors(self):
        c1 = self._make([1.0], ["a"])
        c2 = self._make([1.0], ["b"])
        self.assertFalse(cores_close(c1, c2))


# ── Test: NumpyEinsumContractor optimize flag ─────────────────────────────────

class TestNumpyEinsumOptimize(unittest.TestCase):

    def test_result_unchanged_with_optimize(self):
        """optimize='greedy' must give the same result as no optimization."""
        cores = {
            "ab": NumpyCore(NEQ3.copy(), ["a", "b"], "ab"),
            "bc": NumpyCore(NEQ3.copy(), ["b", "c"], "bc"),
            "ac": NumpyCore(NEQ3.copy(), ["a", "c"], "ac"),
        }
        result_default = NumpyEinsumContractor(
            cores, openColors=["a"], optimize=False
        ).contract()
        result_greedy = NumpyEinsumContractor(
            cores, openColors=["a"], optimize="greedy"
        ).contract()
        np.testing.assert_allclose(result_default.values, result_greedy.values)

    def test_scalar_contraction(self):
        """Z of an AllDiff 3-chain equals 3! = 6."""
        cores = {
            "ab": NumpyCore(NEQ3.copy(), ["a", "b"], "ab"),
            "bc": NumpyCore(NEQ3.copy(), ["b", "c"], "bc"),
            "ac": NumpyCore(NEQ3.copy(), ["a", "c"], "ac"),
        }
        Z = float(engine.contract(cores, openColors=[])[:])
        self.assertAlmostEqual(Z, 6.0)


# ── Test: VariableEliminationContractor ──────────────────────────────────────

class TestVariableElimination(unittest.TestCase):

    def test_chain_scalar(self):
        """VE on a 3-chain NEQ network gives Z = 6."""
        cores = {
            "ab": NumpyCore(NEQ3.copy(), ["a", "b"], "ab"),
            "bc": NumpyCore(NEQ3.copy(), ["b", "c"], "bc"),
            "ac": NumpyCore(NEQ3.copy(), ["a", "c"], "ac"),
        }
        Z = float(engine.contract(cores, openColors=[],
                                  contractionMethod="VariableElimination")[:])
        self.assertAlmostEqual(Z, 6.0)

    def test_chain_marginal(self):
        """VE marginal over 'a' of an unconstrained 3-chain is uniform."""
        cores = {
            "ab": NumpyCore(NEQ3.copy(), ["a", "b"], "ab"),
            "bc": NumpyCore(NEQ3.copy(), ["b", "c"], "bc"),
            "ac": NumpyCore(NEQ3.copy(), ["a", "c"], "ac"),
        }
        result = engine.contract(cores, openColors=["a"],
                                 contractionMethod="VariableElimination")
        vals = result.values / result.values.sum()
        np.testing.assert_allclose(vals, [1/3, 1/3, 1/3], atol=1e-9)

    def test_matches_einsum(self):
        """VE and NumpyEinsum give the same marginals."""
        cores = {
            "ab": NumpyCore(NEQ3.copy(), ["a", "b"], "ab"),
            "bc": NumpyCore(NEQ3.copy(), ["b", "c"], "bc"),
            "ac": NumpyCore(NEQ3.copy(), ["a", "c"], "ac"),
        }
        r_ve  = engine.contract(cores, openColors=["a"],
                                contractionMethod="VariableElimination")
        r_ein = engine.contract(cores, openColors=["a"],
                                contractionMethod="NumpyEinsum")
        v_ve  = r_ve.values / r_ve.values.sum()
        v_ein = r_ein.values / r_ein.values.sum()
        np.testing.assert_allclose(v_ve, v_ein, atol=1e-9)

    def test_existing_factor_key_is_not_overwritten(self):
        """VE-generated keys must not overwrite surviving user factors."""
        cores = {
            "ab": NumpyCore(np.ones((2, 2)), ["a", "b"], "ab"),
            "_ve_0": NumpyCore(np.array([3.0, 5.0]), ["z"], "_ve_0"),
        }
        result = engine.contract(
            cores, openColors=["z"], contractionMethod="VariableElimination"
        )
        self.assertEqual(result.colors, ["z"])
        np.testing.assert_allclose(result.values, np.array([12.0, 20.0]))


# ── Test: ForwardChaining ─────────────────────────────────────────────────────

class TestForwardChaining(unittest.TestCase):

    def test_known_variable_propagates(self):
        """Fixing a=0 propagates: b and c are restricted to {1,2}."""
        cores = {
            "ab": _neq(["a", "b"]),
            "bc": _neq(["b", "c"]),
            "ac": _neq(["a", "c"]),
            "fa": _unit("a", 0),
        }
        fc = ForwardChaining(cores).run()
        self.assertTrue(fc.consistent)
        m_a = fc.marginal("a")
        np.testing.assert_allclose(m_a, [1, 0, 0])
        # b and c must avoid value 0
        m_b = fc.marginal("b")
        m_c = fc.marginal("c")
        self.assertIsNotNone(m_b)
        self.assertAlmostEqual(float(m_b[0]), 0.0, places=6)
        self.assertIsNotNone(m_c)
        self.assertAlmostEqual(float(m_c[0]), 0.0, places=6)

    def test_fully_determined_row(self):
        """Setting all 3 values in a row disentangles everything."""
        cores = {
            "ab": _neq(["a", "b"]),
            "bc": _neq(["b", "c"]),
            "ac": _neq(["a", "c"]),
            "fa": _unit("a", 0),
            "fb": _unit("b", 1),
            "fc": _unit("c", 2),
        }
        fc = ForwardChaining(cores).run()
        self.assertTrue(fc.consistent)
        self.assertTrue(fc.fully_disentangled)
        np.testing.assert_allclose(fc.marginal("a"), [1, 0, 0])
        np.testing.assert_allclose(fc.marginal("b"), [0, 1, 0])
        np.testing.assert_allclose(fc.marginal("c"), [0, 0, 1])

    def test_contradiction_detected(self):
        """Two units with the same value in the same row → inconsistent."""
        cores = {
            "ab": _neq(["a", "b"]),
            "fa": _unit("a", 1),
            "fb": _unit("b", 1),   # contradicts NEQ
        }
        fc = ForwardChaining(cores).run()
        self.assertFalse(fc.consistent)

    def test_3x3_latin_row_determined(self):
        """FC with full row 0 determined resolves c00, c01, c02."""
        cores = _latin3({(0, 0): 0, (0, 1): 1, (0, 2): 2})
        fc = ForwardChaining(cores).run()
        self.assertTrue(fc.consistent)
        self.assertIn("c00", fc.disentangledNodes)
        self.assertIn("c01", fc.disentangledNodes)
        self.assertIn("c02", fc.disentangledNodes)
        np.testing.assert_allclose(fc.marginal("c00"), [1, 0, 0])
        np.testing.assert_allclose(fc.marginal("c01"), [0, 1, 0])
        np.testing.assert_allclose(fc.marginal("c02"), [0, 0, 1])


# ── Test: backtracking ────────────────────────────────────────────────────────

class TestBacktracking(unittest.TestCase):

    def test_3x3_empty_finds_valid_solution(self):
        """Backtracking on an empty 3×3 grid finds a valid Latin square."""
        cores = _latin3()
        success, result, _ = backtrack_with_restarts(
            cores, max_depth=15, max_restarts=30, seed=0
        )
        self.assertTrue(success)
        sol = extract_solution(result)
        self.assertTrue(_is_valid_latin(sol, size=3))

    def test_3x3_with_givens_respects_clues(self):
        """Solution must agree with the given clues."""
        givens = {(0, 0): 0, (1, 1): 1, (2, 2): 2}
        cores = _latin3(givens)
        success, result, _ = backtrack_with_restarts(
            cores, max_depth=15, max_restarts=50, seed=42
        )
        self.assertTrue(success)
        sol = extract_solution(result)
        self.assertTrue(_is_valid_latin(sol, size=3))
        for (r, c), v in givens.items():
            self.assertEqual(sol.get(f"c{r}{c}"), v,
                             msg=f"Given at ({r},{c}) not respected")

    def test_3x3_contradiction_returns_failure(self):
        """Two equal values in the same row → no solution."""
        # Force c00 = c01 = 0, which contradicts row NEQ
        cores = _latin3({(0, 0): 0, (0, 1): 0})
        success, result, _ = backtrack_with_restarts(
            cores, max_depth=10, max_restarts=5, seed=0
        )
        self.assertFalse(success)

    def test_restarts_improve_success_rate(self):
        """With enough restarts, hard instances are solved."""
        # Minimal givens — requires search
        cores = _latin3({(0, 0): 2})
        success, result, restarts = backtrack_with_restarts(
            cores, max_depth=15, max_restarts=100, seed=7
        )
        self.assertTrue(success)
        sol = extract_solution(result)
        self.assertTrue(_is_valid_latin(sol))

    def test_extract_solution_gives_valid_values(self):
        """extract_solution values are in range [0, D)."""
        cores = _latin3()
        success, result, _ = backtrack_with_restarts(
            cores, max_depth=15, max_restarts=30, seed=1
        )
        self.assertTrue(success)
        sol = extract_solution(result)
        for v in sol.values():
            self.assertGreaterEqual(v, 0)
            self.assertLess(v, D)

    def test_single_backtrack_call(self):
        """Direct backtrack() works on a trivial case."""
        cores = {
            "ab": _neq(["a", "b"]),
            "fa": _unit("a", 0),
        }
        success, result = backtrack(cores, max_depth=5)
        self.assertTrue(success)
        sol = extract_solution(result)
        self.assertEqual(sol.get("a"), 0)
        self.assertNotEqual(sol.get("b"), 0)


# ── Test: Semiring parameterisation ──────────────────────────────────────────

class TestSemirings(unittest.TestCase):
    """Four-semiring evaluation on the same NEQ chain (domain {0,1})."""

    def setUp(self):
        from tnreason.engine.semirings import SUM_PRODUCT, MAX_PRODUCT, BOOLEAN, TROPICAL
        self.SUM_PRODUCT = SUM_PRODUCT
        self.MAX_PRODUCT = MAX_PRODUCT
        self.BOOLEAN     = BOOLEAN
        self.TROPICAL    = TROPICAL

        NEQ  = np.array([[0., 1.], [1., 0.]])
        P0   = np.array([0.3, 0.7])   # prior: X0=1 is more likely
        self.cores = {
            'n01': NumpyCore(NEQ.copy(), ['x0','x1'], 'n01'),
            'n12': NumpyCore(NEQ.copy(), ['x1','x2'], 'n12'),
            'n23': NumpyCore(NEQ.copy(), ['x2','x3'], 'n23'),
            'p0' : NumpyCore(P0.copy(),  ['x0'],      'p0'),
        }
        neq_cost = np.where(NEQ == 0, np.inf, 0.)
        self.tcores = {
            'n01': NumpyCore(neq_cost.copy(), ['x0','x1'], 'n01'),
            'n12': NumpyCore(neq_cost.copy(), ['x1','x2'], 'n12'),
            'n23': NumpyCore(neq_cost.copy(), ['x2','x3'], 'n23'),
            'c0' : NumpyCore(-np.log(P0),     ['x0'],      'c0'),
        }

    def test_boolean_satisfiable(self):
        """Boolean semiring: NEQ chain with {0,1} cores is satisfiable → Z = 1."""
        NEQ = np.array([[0., 1.], [1., 0.]])
        bool_cores = {
            'n01': NumpyCore(NEQ.copy(), ['x0','x1'], 'n01'),
            'n12': NumpyCore(NEQ.copy(), ['x1','x2'], 'n12'),
            'n23': NumpyCore(NEQ.copy(), ['x2','x3'], 'n23'),
        }
        Z = float(engine.contract(bool_cores, openColors=[], semiring=self.BOOLEAN)[:])
        self.assertAlmostEqual(Z, 1.0)

    def test_boolean_unsatisfiable(self):
        """Boolean semiring: contradictory givens → Z = 0."""
        cores = {
            'n01': NumpyCore(np.array([[0.,1.],[1.,0.]]), ['x0','x1'], 'n'),
            'fx0': NumpyCore(np.array([1., 0.]), ['x0'], 'fx0'),
            'fx1': NumpyCore(np.array([1., 0.]), ['x1'], 'fx1'),  # x0=x1=0 violates NEQ
        }
        Z = float(engine.contract(cores, openColors=[], semiring=self.BOOLEAN)[:])
        self.assertAlmostEqual(Z, 0.0)

    def test_sum_product_matches_einsum(self):
        """SumProduct semiring must give same marginals as NumpyEinsum."""
        r_sr  = engine.contract(self.cores, openColors=['x3'], semiring=self.SUM_PRODUCT)
        r_ein = engine.contract(self.cores, openColors=['x3'])
        m_sr  = r_sr.values  / r_sr.values.sum()
        m_ein = r_ein.values / r_ein.values.sum()
        np.testing.assert_allclose(m_sr, m_ein, atol=1e-9)

    def test_sum_product_marginal_value(self):
        """P(X3=1) should equal P(X1=1) by the alternating NEQ chain."""
        r = engine.contract(self.cores, openColors=['x3'], semiring=self.SUM_PRODUCT)
        marg = r.values / r.values.sum()
        # With P0=[0.3,0.7]: chain alternates, X3 has same distribution as X1
        self.assertAlmostEqual(float(marg[1]), 0.3, places=5)

    def test_max_product_map_alternates(self):
        """MaxProduct MAP should alternate 1,0,1,0 (X0 prefers 1, rest forced)."""
        r = engine.contract(
            self.cores, openColors=['x0','x1','x2','x3'], semiring=self.MAX_PRODUCT
        )
        idx = np.unravel_index(np.argmax(r.values), r.values.shape)
        # x0=1 (highest prior), x1=0, x2=1, x3=0
        colors = r.colors
        assignment = dict(zip(colors, [int(i) for i in idx]))
        self.assertEqual(assignment['x0'], 1)
        self.assertEqual(assignment['x1'], 0)

    def test_tropical_min_cost(self):
        """Tropical semiring: minimum-cost X3 matches Max-Product MAP for X3."""
        r = engine.contract(self.tcores, openColors=['x3'], semiring=self.TROPICAL)
        best_x3 = int(np.argmin(r.values))
        # Max-Product MAP has x3=0 (alternating from x0=1)
        self.assertEqual(best_x3, 0)

    def test_semiring_registry(self):
        """get_semiring returns the correct semiring by name."""
        from tnreason.engine.semirings import get_semiring, SUM_PRODUCT
        sr = get_semiring("SumProduct")
        self.assertIs(sr, SUM_PRODUCT)

    def test_unknown_semiring_raises(self):
        from tnreason.engine.semirings import get_semiring
        with self.assertRaises(ValueError):
            get_semiring("NonExistentSemiring")


if __name__ == "__main__":
    unittest.main()
