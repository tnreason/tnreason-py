import unittest

import numpy as np

from experiments.constraint_networks import constraint_networks as cn
from tnreason import engine
from tnreason.engine.workload_to_numpy import NumpyCore


class CoreComparisonTest(unittest.TestCase):
    def test_exact_equal_matches_colors_by_name(self):
        core0 = NumpyCore(
            values=np.array([[1, 2, 3], [4, 5, 6]]),
            colors=["a", "b"],
            name="core0",
        )
        core1 = NumpyCore(
            values=np.array([[1, 4], [2, 5], [3, 6]]),
            colors=["b", "a"],
            name="core1",
        )

        self.assertTrue(engine.cores_equal(core0, core1))
        self.assertTrue(core0 == core1)
        self.assertTrue(core0.eq_old(core1))

    def test_exact_equal_rejects_small_float_difference(self):
        core0 = NumpyCore(values=np.array([1.0, 2.0]), colors=["a"], name="core0")
        core1 = NumpyCore(values=np.array([1.0, 2.0 + 1e-10]), colors=["a"], name="core1")

        self.assertFalse(engine.cores_equal(core0, core1))
        self.assertFalse(core0 == core1)
        self.assertTrue(engine.cores_close(core0, core1))

    def test_comparison_rejects_named_shape_mismatch(self):
        core0 = NumpyCore(values=np.zeros((2, 3)), colors=["a", "b"], name="core0")
        core1 = NumpyCore(values=np.zeros((2, 3)), colors=["b", "a"], name="core1")

        self.assertFalse(engine.cores_equal(core0, core1))
        self.assertFalse(engine.cores_close(core0, core1))

    def test_non_numpy_cores_use_fallback_comparison(self):
        core0 = engine.create_from_slice_iterator(
            shape=[2, 3],
            colors=["a", "b"],
            sliceIterator=[(2, {"a": 1, "b": 2})],
            coreType="PolynomialCore",
        )
        core1 = engine.create_from_slice_iterator(
            shape=[3, 2],
            colors=["b", "a"],
            sliceIterator=[(2, {"a": 1, "b": 2})],
            coreType="PolynomialCore",
        )

        self.assertTrue(engine.cores_equal(core0, core1))
        self.assertTrue(core0 == core1)

    def test_split_edge_accepts_floating_reconstruction_noise(self):
        values = np.outer(np.array([0.1, 0.2]), np.array([0.3, 0.4]))
        core = engine.create_from_slice_iterator(
            colors=["a", "b"],
            shape=[2, 2],
            sliceIterator=[
                (values[index], {"a": index[0], "b": index[1]})
                for index in np.ndindex(values.shape)
            ],
        )

        con_net = cn.ConstraintNetwork(coresDict={"factor": core})

        self.assertTrue(con_net.split_edge("factor", colors0=["a"], colors1=["b"]))
        self.assertEqual(set(con_net.coresDict), {"factor_0", "factor_1"})


if __name__ == "__main__":
    unittest.main()
