import unittest

import numpy as np

from tnreason.logic import LogicCompiler, TTGadgets, WorldModelUtils, consistency_certificate


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


class LogicModuleTest(unittest.TestCase):
    def test_empty_system_returns_unit_partition(self):
        self.assertEqual(LogicCompiler(domain_size=3).solve(), 1.0)

    def test_unconstrained_variables_count_all_assignments(self):
        compiler = LogicCompiler(domain_size=3)
        compiler.add_variable("x")
        compiler.add_variable("y")
        self.assertEqual(compiler.solve(), 9.0)

    def test_compile_rejects_too_many_variables(self):
        compiler = LogicCompiler(domain_size=2)
        for i in range(63):
            compiler.add_variable(f"v{i}")
        with self.assertRaises(ValueError):
            compiler.compile()

    def test_custom_tensor_shape_is_validated(self):
        compiler = LogicCompiler(domain_size=2)
        with self.assertRaises(ValueError):
            compiler.add_constraint("CUSTOM", ("x", "y"), tensor=np.ones((2, 2, 2)))

    def test_unknown_semiring_raises(self):
        compiler = LogicCompiler(domain_size=2)
        compiler.add_constraint("EQ", ("x", "y"))
        with self.assertRaises(ValueError):
            compiler.solve("mystery")

    def test_gt_constraint_counts_strict_orderings(self):
        compiler = LogicCompiler(domain_size=4)
        compiler.add_constraint("GT", ("x", "y"))
        self.assertEqual(compiler.solve(), 6.0)

    def test_alldiff_contract_counts_injective_assignments(self):
        cores = TTGadgets.alldiff_cores(2, 3)
        actual_assignments = {(): {0}}
        for core in cores:
            num_states, source_masks, target_masks = core
            next_assignments = {}
            for source_mask, target_mask in zip(source_masks, target_masks):
                for assignment, reachable_states in actual_assignments.items():
                    if source_mask not in reachable_states:
                        continue
                    changed_bit = int(target_mask ^ source_mask)
                    self.assertTrue(is_power_of_two(changed_bit))
                    value = changed_bit.bit_length() - 1
                    next_assignments.setdefault(assignment + (value,), set()).add(target_mask)
            actual_assignments = next_assignments
        self.assertEqual(
            set(actual_assignments),
            {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)},
        )
        self.assertEqual(TTGadgets.contract_tt(cores), 6.0)
        self.assertEqual(TTGadgets.contract_tt(TTGadgets.alldiff_cores(4, 3)), 0.0)

    def test_build_factored_transition_shapes_and_moves(self):
        cores = WorldModelUtils.build_factored_transition(3)
        self.assertEqual(len(cores), 2)
        self.assertEqual(cores[0].shape, (3, 3, 4))
        self.assertEqual(cores[0][0, 0, 0], 1.0)
        self.assertEqual(cores[1][1, 0, 1], 1.0)

    def test_build_factored_transition_rejects_unsupported_dims(self):
        with self.assertRaises(ValueError):
            WorldModelUtils.build_factored_transition(3, dims=3)

    def test_consistency_certificate_is_clamped(self):
        self.assertEqual(consistency_certificate(-1.0, 2.0), 0.0)
        self.assertEqual(consistency_certificate(3.0, 2.0), 1.0)
        with self.assertRaises(ValueError):
            consistency_certificate(1.0, 0.0)


if __name__ == "__main__":
    unittest.main()
