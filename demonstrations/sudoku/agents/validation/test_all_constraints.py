"""
Complete Sudoku Constraint Validation Suite

This module validates ALL sudoku constraints by testing that tensor implementations
correctly enforce their natural language rules.

Coverage:
- NEW constraints (agents/): Anti-Knight, German Whispers, Renban
- EXISTING constraints (constraints/): Red Line, Black Dot, White Dot

Run with:
    python validation/test_all_constraints.py
"""

import unittest
import sys
from pathlib import Path

# Setup paths
parent_dir = str(Path(__file__).parent.parent)
project_root = str(Path(__file__).parent.parent.parent.parent)
constraints_dir = str(Path(__file__).parent.parent.parent / "constraints")

for path in [parent_dir, project_root, constraints_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import new constraints
from constraints.knight_move import (
    knight_move_constraint, cenc_knight_move_constraint,
    get_knight_moves_for_cell, generate_all_knight_constraints
)
from constraints.german_whispers import (
    german_whispers_constraint, cenc_german_whispers_constraint,
    generate_german_whispers_constraints
)
from constraints.renban import (
    renban_constraint, cenc_renban_constraint,
    generate_renban_constraints
)

# Import existing constraints
from red_line import (
    red_line_constraint, prepare_odd_indicator_core,
    red_line_constraint_from_odd, benc_prepare_odd_indicator_core,
    cenc_red_line_constraint
)
from black_dot import black_dot_constraint, cenc_black_dot_constraint
from white_dot import white_dot_constraint, cenc_white_dot_constraint


# ============================================================================
# ANTI-KNIGHT CONSTRAINT TESTS
# ============================================================================

class TestKnightMoveConstraint(unittest.TestCase):
    """
    Anti-Knight: "Digits at knight's move positions cannot be identical."
    """
    
    def test_knight_allows_different_values(self):
        """Knight constraint should allow different values."""
        self.assertEqual(knight_move_constraint("p1", "p2")[{"p1": 1, "p2": 2}], 1)
        self.assertEqual(knight_move_constraint("p1", "p2")[{"p1": 0, "p2": 8}], 1)
    
    def test_knight_rejects_same_values(self):
        """Knight constraint should reject identical values."""
        self.assertEqual(knight_move_constraint("p1", "p2")[{"p1": 5, "p2": 5}], 0)
        self.assertEqual(knight_move_constraint("p1", "p2")[{"p1": 0, "p2": 0}], 0)
    
    def test_knight_categorical_encoding(self):
        """Test categorical encoding version."""
        self.assertEqual(cenc_knight_move_constraint("p1", "p2")[{"p1": 1, "p2": 2}], 1)
        self.assertEqual(cenc_knight_move_constraint("p1", "p2")[{"p1": 5, "p2": 5}], 0)
    
    def test_knight_move_geometry_center(self):
        """Center cell (4,4) should have 8 knight moves."""
        moves = get_knight_moves_for_cell(4, 4)
        self.assertEqual(len(moves), 8)
        self.assertIn((2, 3), moves)
        self.assertIn((6, 5), moves)
    
    def test_knight_move_geometry_corner(self):
        """Corner cell (0,0) should have 2 knight moves."""
        moves = get_knight_moves_for_cell(0, 0)
        self.assertEqual(len(moves), 2)
        self.assertIn((1, 2), moves)
        self.assertIn((2, 1), moves)
    
    def test_knight_move_geometry_edge(self):
        """Edge cell (0,4) should have 4 knight moves."""
        moves = get_knight_moves_for_cell(0, 4)
        self.assertEqual(len(moves), 4)
    
    def test_knight_move_boundary_conditions(self):
        """Test that knight moves don't go off the board."""
        for row, col in [(0,0), (0,8), (8,0), (8,8)]:
            moves = get_knight_moves_for_cell(row, col)
            for r, c in moves:
                self.assertTrue(0 <= r < 9 and 0 <= c < 9)
    
    def test_generate_all_knight_constraints(self):
        """Test generation of all knight constraints."""
        constraints = generate_all_knight_constraints(grid_size=9)
        self.assertGreater(len(constraints), 100)
        for name in constraints:
            self.assertTrue(name.startswith("knight_"))


# ============================================================================
# GERMAN WHISPERS CONSTRAINT TESTS
# ============================================================================

class TestGermanWhispersConstraint(unittest.TestCase):
    """
    German Whispers: "Adjacent digits along a line must differ by at least 5."
    """
    
    def test_german_whispers_valid_pairs(self):
        """Should allow pairs with difference >= 5."""
        self.assertEqual(german_whispers_constraint("p1", "p2")[{"p1": 0, "p2": 5}], 1)  # 1,6
        self.assertEqual(german_whispers_constraint("p1", "p2")[{"p1": 0, "p2": 8}], 1)  # 1,9
        self.assertEqual(german_whispers_constraint("p1", "p2")[{"p1": 3, "p2": 8}], 1)  # 4,9
    
    def test_german_whispers_invalid_pairs(self):
        """Should reject pairs with difference < 5."""
        self.assertEqual(german_whispers_constraint("p1", "p2")[{"p1": 0, "p2": 4}], 0)  # 1,5
        self.assertEqual(german_whispers_constraint("p1", "p2")[{"p1": 4, "p2": 5}], 0)  # 5,6
        self.assertEqual(german_whispers_constraint("p1", "p2")[{"p1": 4, "p2": 4}], 0)  # same
    
    def test_german_whispers_boundary_case(self):
        """Exactly difference of 5 should be valid."""
        self.assertEqual(german_whispers_constraint("p1", "p2")[{"p1": 0, "p2": 5}], 1)
        self.assertEqual(german_whispers_constraint("p1", "p2")[{"p1": 1, "p2": 6}], 1)
        self.assertEqual(german_whispers_constraint("p1", "p2")[{"p1": 3, "p2": 8}], 1)
    
    def test_german_whispers_categorical_encoding(self):
        """Test categorical encoding version."""
        self.assertEqual(cenc_german_whispers_constraint("p1", "p2")[{"p1": 0, "p2": 5}], 1)
        self.assertEqual(cenc_german_whispers_constraint("p1", "p2")[{"p1": 0, "p2": 4}], 0)
    
    def test_german_whispers_line_generation(self):
        """Test constraint generation for a line."""
        line = [(0, 0), (0, 1), (0, 2), (0, 3)]
        constraints = generate_german_whispers_constraints(line, "test_gw")
        self.assertEqual(len(constraints), 3)
    
    def test_german_whispers_exhaustive_validation(self):
        """Exhaustively test all pairs."""
        for v1 in range(9):
            for v2 in range(9):
                result = german_whispers_constraint("p1", "p2")[{"p1": v1, "p2": v2}]
                expected = 1 if abs((v1 + 1) - (v2 + 1)) >= 5 else 0
                self.assertEqual(result, expected)


# ============================================================================
# RENBAN CONSTRAINT TESTS
# ============================================================================

class TestRenbanConstraint(unittest.TestCase):
    """
    Renban: "Line contains consecutive non-repeating digits in any order."
    """
    
    def test_renban_valid_consecutive(self):
        """Should allow consecutive non-repeating digits."""
        pos_vars = ["p1", "p2", "p3"]
        self.assertEqual(renban_constraint(pos_vars)[{"p1": 0, "p2": 1, "p3": 2}], 1)  # 1,2,3
        self.assertEqual(renban_constraint(pos_vars)[{"p1": 3, "p2": 4, "p3": 5}], 1)  # 4,5,6
    
    def test_renban_valid_different_order(self):
        """Should allow different orderings of same set."""
        pos_vars = ["p1", "p2", "p3"]
        self.assertEqual(renban_constraint(pos_vars)[{"p1": 2, "p2": 0, "p3": 1}], 1)  # 3,1,2
        self.assertEqual(renban_constraint(pos_vars)[{"p1": 8, "p2": 6, "p3": 7}], 1)  # 9,7,8
    
    def test_renban_invalid_not_consecutive(self):
        """Should reject non-consecutive sets."""
        pos_vars = ["p1", "p2", "p3"]
        self.assertEqual(renban_constraint(pos_vars)[{"p1": 0, "p2": 1, "p3": 3}], 0)  # 1,2,4
        self.assertEqual(renban_constraint(pos_vars)[{"p1": 0, "p2": 2, "p3": 4}], 0)  # 1,3,5
    
    def test_renban_invalid_repeating(self):
        """Should reject repeating digits."""
        pos_vars = ["p1", "p2", "p3"]
        self.assertEqual(renban_constraint(pos_vars)[{"p1": 1, "p2": 1, "p3": 2}], 0)
        self.assertEqual(renban_constraint(pos_vars)[{"p1": 4, "p2": 4, "p3": 4}], 0)
    
    def test_renban_two_positions(self):
        """Test with 2 positions."""
        pos_vars = ["p1", "p2"]
        self.assertEqual(renban_constraint(pos_vars)[{"p1": 0, "p2": 1}], 1)  # 1,2
        self.assertEqual(renban_constraint(pos_vars)[{"p1": 0, "p2": 2}], 0)  # 1,3
    
    def test_renban_four_positions(self):
        """Test with 4 positions."""
        pos_vars = ["p1", "p2", "p3", "p4"]
        self.assertEqual(renban_constraint(pos_vars)[{"p1": 0, "p2": 1, "p3": 2, "p4": 3}], 1)
        self.assertEqual(renban_constraint(pos_vars)[{"p1": 0, "p2": 1, "p3": 2, "p4": 4}], 0)
    
    def test_renban_categorical_encoding(self):
        """Test categorical encoding version."""
        pos_vars = ["p1", "p2", "p3"]
        self.assertEqual(cenc_renban_constraint(pos_vars)[{"p1": 0, "p2": 1, "p3": 2}], 1)
        self.assertEqual(cenc_renban_constraint(pos_vars)[{"p1": 0, "p2": 1, "p3": 3}], 0)
    
    def test_renban_line_generation(self):
        """Test constraint generation for a line."""
        line = [(0, 0), (0, 1), (0, 2)]
        constraints = generate_renban_constraints(line, "test_renban")
        self.assertEqual(len(constraints), 1)
        self.assertIn("test_renban", constraints)
    
    def test_renban_exhaustive_2position(self):
        """Exhaustively test all pairs for 2-position renban."""
        pos_vars = ["p1", "p2"]
        for v1 in range(9):
            for v2 in range(9):
                result = renban_constraint(pos_vars)[{"p1": v1, "p2": v2}]
                expected = 1 if abs(v1 - v2) == 1 else 0
                self.assertEqual(result, expected)


# ============================================================================
# RED LINE CONSTRAINT TESTS
# ============================================================================

class TestRedLineConstraint(unittest.TestCase):
    """
    Red Line: "Exactly one of the neighbored positions is odd."
    """
    
    def test_odd_indicator_even_value(self):
        """Odd indicator should correctly identify odd/even."""
        # Value 0 (digit 1, odd) -> odd indicator = 1
        self.assertEqual(prepare_odd_indicator_core("pos", "odd")[{"pos": 0, "odd": 1}], 1)
        # Value 1 (digit 2, even) -> odd indicator = 0
        self.assertEqual(prepare_odd_indicator_core("pos", "odd")[{"pos": 1, "odd": 0}], 1)
        # Value 3 (digit 4, even) -> odd indicator = 0
        self.assertEqual(prepare_odd_indicator_core("pos", "odd")[{"pos": 3, "odd": 0}], 1)
    
    def test_odd_indicator_basis_encoding(self):
        """Test basis encoding version."""
        self.assertEqual(benc_prepare_odd_indicator_core("pos", "odd")[{"pos": 3, "odd": 0}], 1)
        self.assertEqual(benc_prepare_odd_indicator_core("pos", "odd")[{"pos": 3, "odd": 1}], 0)
    
    def test_red_line_from_odd_valid(self):
        """Should accept exactly one odd (XOR)."""
        self.assertEqual(red_line_constraint_from_odd("o1", "o2")[{"o1": 1, "o2": 0}], 1)
        self.assertEqual(red_line_constraint_from_odd("o1", "o2")[{"o1": 0, "o2": 1}], 1)
    
    def test_red_line_from_odd_invalid(self):
        """Should reject both odd or both even."""
        self.assertEqual(red_line_constraint_from_odd("o1", "o2")[{"o1": 1, "o2": 1}], 0)
        self.assertEqual(red_line_constraint_from_odd("o1", "o2")[{"o1": 0, "o2": 0}], 0)
    
    def test_red_line_categorical_encoding(self):
        """Test categorical encoding (XOR)."""
        self.assertEqual(cenc_red_line_constraint("o1", "o2")[{"o1": 1, "o2": 0}], 1)
        self.assertEqual(cenc_red_line_constraint("o1", "o2")[{"o1": 1, "o2": 1}], 0)
    
    def test_red_line_full_constraint(self):
        """Test full constraint with position variables."""
        constraint_dict = red_line_constraint("pos1", "pos2")
        self.assertIn("oddInd_pos1_core", constraint_dict)
        self.assertIn("red_line_pos1_pos2", constraint_dict)
        
        red_line_core = constraint_dict["red_line_pos1_pos2"]
        self.assertEqual(red_line_core[{"oddInd_pos1": 0, "oddInd_pos2": 1}], 1)
        self.assertEqual(red_line_core[{"oddInd_pos1": 1, "oddInd_pos2": 0}], 1)
        self.assertEqual(red_line_core[{"oddInd_pos1": 0, "oddInd_pos2": 0}], 0)
    
    def test_red_line_natural_language_interpretation(self):
        """Test with natural language interpretation."""
        for digit in range(1, 10):
            value = digit - 1
            expected_odd = digit % 2
            result = prepare_odd_indicator_core("pos", "odd")[{"pos": value, "odd": expected_odd}]
            self.assertEqual(result, 1)


# ============================================================================
# BLACK DOT CONSTRAINT TESTS
# ============================================================================

class TestBlackDotConstraint(unittest.TestCase):
    """
    Black Dot: "Digits separated by a black dot are in a 1:2 ratio."
    """
    
    def test_black_dot_valid_ratio_pairs(self):
        """Should accept 1:2 ratio pairs."""
        self.assertEqual(black_dot_constraint("p1", "p2")[{"p1": 0, "p2": 1}], 1)  # 1,2
        self.assertEqual(black_dot_constraint("p1", "p2")[{"p1": 1, "p2": 3}], 1)  # 2,4
        self.assertEqual(black_dot_constraint("p1", "p2")[{"p1": 2, "p2": 5}], 1)  # 3,6
        self.assertEqual(black_dot_constraint("p1", "p2")[{"p1": 3, "p2": 7}], 1)  # 4,8
    
    def test_black_dot_invalid_ratio_pairs(self):
        """Should reject non-1:2 ratio pairs."""
        self.assertEqual(black_dot_constraint("p1", "p2")[{"p1": 0, "p2": 2}], 0)  # 1,3
        self.assertEqual(black_dot_constraint("p1", "p2")[{"p1": 1, "p2": 2}], 0)  # 2,3
        self.assertEqual(black_dot_constraint("p1", "p2")[{"p1": 4, "p2": 5}], 0)  # 5,6
    
    def test_black_dot_same_value(self):
        """Should reject same values."""
        for v in range(9):
            self.assertEqual(black_dot_constraint("p1", "p2")[{"p1": v, "p2": v}], 0)
    
    def test_black_dot_categorical_encoding(self):
        """Test categorical encoding version."""
        self.assertEqual(cenc_black_dot_constraint("p1", "p2")[{"p1": 0, "p2": 1}], 1)
        self.assertEqual(cenc_black_dot_constraint("p1", "p2")[{"p1": 2, "p2": 5}], 1)
        self.assertEqual(cenc_black_dot_constraint("p1", "p2")[{"p1": 0, "p2": 2}], 0)
    
    def test_black_dot_exhaustive_validation(self):
        """Exhaustively test all pairs."""
        for v1 in range(9):
            for v2 in range(9):
                result = black_dot_constraint("p1", "p2")[{"p1": v1, "p2": v2}]
                d1, d2 = v1 + 1, v2 + 1
                expected = 1 if (d1 == 2 * d2 or d2 == 2 * d1) else 0
                self.assertEqual(result, expected)
    
    def test_black_dot_edge_cases(self):
        """Test edge cases."""
        # Digit 5 has no valid black dot pairs (5*2=10 out of range, 5/2=2.5 not integer)
        for v2 in range(9):
            self.assertEqual(black_dot_constraint("p1", "p2")[{"p1": 4, "p2": v2}], 0)
        # Digit 9 has no valid pairs
        for v2 in range(9):
            self.assertEqual(black_dot_constraint("p1", "p2")[{"p1": 8, "p2": v2}], 0)


# ============================================================================
# WHITE DOT CONSTRAINT TESTS
# ============================================================================

class TestWhiteDotConstraint(unittest.TestCase):
    """
    White Dot: "Cells joined by a white dot contain consecutive digits."
    """
    
    def test_white_dot_valid_consecutive_pairs(self):
        """Should accept consecutive pairs."""
        self.assertEqual(white_dot_constraint("p1", "p2")[{"p1": 0, "p2": 1}], 1)  # 1,2
        self.assertEqual(white_dot_constraint("p1", "p2")[{"p1": 1, "p2": 2}], 1)  # 2,3
        self.assertEqual(white_dot_constraint("p1", "p2")[{"p1": 4, "p2": 5}], 1)  # 5,6
        self.assertEqual(white_dot_constraint("p1", "p2")[{"p1": 7, "p2": 8}], 1)  # 8,9
    
    def test_white_dot_invalid_non_consecutive_pairs(self):
        """Should reject non-consecutive pairs."""
        self.assertEqual(white_dot_constraint("p1", "p2")[{"p1": 0, "p2": 2}], 0)  # 1,3
        self.assertEqual(white_dot_constraint("p1", "p2")[{"p1": 0, "p2": 8}], 0)  # 1,9
        self.assertEqual(white_dot_constraint("p1", "p2")[{"p1": 3, "p2": 6}], 0)  # 4,7
    
    def test_white_dot_same_value(self):
        """Should reject same values."""
        for v in range(9):
            self.assertEqual(white_dot_constraint("p1", "p2")[{"p1": v, "p2": v}], 0)
    
    def test_white_dot_categorical_encoding(self):
        """Test categorical encoding version."""
        self.assertEqual(cenc_white_dot_constraint("p1", "p2")[{"p1": 0, "p2": 1}], 1)
        self.assertEqual(cenc_white_dot_constraint("p1", "p2")[{"p1": 0, "p2": 2}], 0)
    
    def test_white_dot_exhaustive_validation(self):
        """Exhaustively test all pairs."""
        for v1 in range(9):
            for v2 in range(9):
                result = white_dot_constraint("p1", "p2")[{"p1": v1, "p2": v2}]
                expected = 1 if abs(v1 - v2) == 1 else 0
                self.assertEqual(result, expected)
    
    def test_white_dot_boundary_cases(self):
        """Test boundary cases."""
        # 1 only consecutive with 2
        for v2 in range(2, 9):
            self.assertEqual(white_dot_constraint("p1", "p2")[{"p1": 0, "p2": v2}], 0)
        # 9 only consecutive with 8
        for v2 in range(7):
            self.assertEqual(white_dot_constraint("p1", "p2")[{"p1": 8, "p2": v2}], 0)


# ============================================================================
# CONSTRAINT COMPARISON TESTS
# ============================================================================

class TestConstraintComparison(unittest.TestCase):
    """Tests comparing different constraint types."""
    
    def test_black_vs_white_dot_distinction(self):
        """Black dot (1:2) and white dot (consecutive) should differ."""
        # (1,2): both valid
        self.assertEqual(white_dot_constraint("p1", "p2")[{"p1": 0, "p2": 1}], 1)
        self.assertEqual(black_dot_constraint("p1", "p2")[{"p1": 0, "p2": 1}], 1)
        
        # (2,3): only white valid
        self.assertEqual(white_dot_constraint("p1", "p2")[{"p1": 1, "p2": 2}], 1)
        self.assertEqual(black_dot_constraint("p1", "p2")[{"p1": 1, "p2": 2}], 0)
        
        # (2,4): only black valid
        self.assertEqual(white_dot_constraint("p1", "p2")[{"p1": 1, "p2": 3}], 0)
        self.assertEqual(black_dot_constraint("p1", "p2")[{"p1": 1, "p2": 3}], 1)
    
    def test_all_dots_comprehensive(self):
        """Test that all dot types have distinct valid pairs."""
        only_white, only_black, both_valid, neither = [], [], [], []
        
        for v1 in range(9):
            for v2 in range(9):
                white = white_dot_constraint("p1", "p2")[{"p1": v1, "p2": v2}] == 1
                black = black_dot_constraint("p1", "p2")[{"p1": v1, "p2": v2}] == 1
                
                if white and not black:
                    only_white.append((v1+1, v2+1))
                elif black and not white:
                    only_black.append((v1+1, v2+1))
                elif white and black:
                    both_valid.append((v1+1, v2+1))
                else:
                    neither.append((v1+1, v2+1))
        
        self.assertGreater(len(only_white), 0)
        self.assertGreater(len(only_black), 0)
        self.assertGreater(len(both_valid), 0)
        self.assertGreater(len(neither), 0)
    
    def test_knight_and_german_whispers_compatibility(self):
        """Knight and German whispers should be compatible."""
        knight_result = knight_move_constraint("p1", "p2")[{"p1": 0, "p2": 5}]
        gw_result = german_whispers_constraint("p1", "p2")[{"p1": 0, "p2": 5}]
        self.assertEqual(knight_result, 1)
        self.assertEqual(gw_result, 1)
    
    def test_renban_and_german_whispers_interaction(self):
        """Renban and German whispers are incompatible for consecutive pairs."""
        pos_vars = ["p1", "p2"]
        renban_result = renban_constraint(pos_vars)[{"p1": 0, "p2": 1}]
        gw_result = german_whispers_constraint("p1", "p2")[{"p1": 0, "p2": 1}]
        self.assertEqual(renban_result, 1)  # (1,2) consecutive
        self.assertEqual(gw_result, 0)  # difference only 1


# ============================================================================
# MAIN
# ============================================================================

def run_all_tests():
    """Run all constraint validation tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestKnightMoveConstraint))
    suite.addTests(loader.loadTestsFromTestCase(TestGermanWhispersConstraint))
    suite.addTests(loader.loadTestsFromTestCase(TestRenbanConstraint))
    suite.addTests(loader.loadTestsFromTestCase(TestRedLineConstraint))
    suite.addTests(loader.loadTestsFromTestCase(TestBlackDotConstraint))
    suite.addTests(loader.loadTestsFromTestCase(TestWhiteDotConstraint))
    suite.addTests(loader.loadTestsFromTestCase(TestConstraintComparison))
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    print("=" * 80)
    print(" " * 20 + "SUDOKU CONSTRAINT VALIDATION SUITE")
    print("=" * 80)
    print()
    print("Validating all sudoku constraints:")
    print()
    print("  NEW (agents/):")
    print("    • Anti-Knight: knight's move positions cannot have same value")
    print("    • German Whispers: adjacent digits differ by >= 5")
    print("    • Renban: consecutive non-repeating digits on a line")
    print()
    print("  EXISTING (constraints/):")
    print("    • Red Line: exactly one position is odd")
    print("    • Black Dot: 1:2 ratio")
    print("    • White Dot: consecutive digits")
    print()
    print("=" * 80)
    print()
    
    result = run_all_tests()
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print()
    
    if result.wasSuccessful():
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nAll constraints correctly implement their natural language rules.")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        if result.failures:
            print("\nFailures:")
            for test, _ in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("\nErrors:")
            for test, _ in result.errors:
                print(f"  - {test}")
    
    print("=" * 80)
    sys.exit(0 if result.wasSuccessful() else 1)
