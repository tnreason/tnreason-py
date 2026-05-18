from demonstrations.sudoku.constraints.all_different import all_different_constraint, all_different_pair_constraints
from demonstrations.sudoku.constraints.anti_king import anti_king_constraint, all_anti_king_constraints
from demonstrations.sudoku.constraints.anti_knight import anti_knight_constraint, all_anti_knight_constraints
from demonstrations.sudoku.constraints.arrow_sum import arrow_sum_constraint, arrow_sum_constraints
from demonstrations.sudoku.constraints.balanced_line import balanced_four_cell_line_constraint
from demonstrations.sudoku.constraints.black_dot import black_dot_constraint
from demonstrations.sudoku.constraints.diagonal_product import (
    diagonal_product_constraint,
    diagonal_product_to_hidden_constraint,
    same_product_constraints,
)
from demonstrations.sudoku.constraints.entropic_line import entropic_line_constraints, entropic_window_constraint
from demonstrations.sudoku.constraints.fibonacci_line import fibonacci_line_constraint
from demonstrations.sudoku.constraints.gray_dot import gray_dot_constraint
from demonstrations.sudoku.constraints.inequality import (
    greater_than_constraint,
    inequality_constraint,
    less_than_constraint,
)
from demonstrations.sudoku.constraints.killer_cage import killer_cage_constraint, killer_cage_constraints
from demonstrations.sudoku.constraints.little_killer import (
    little_killer_constraint,
    little_killer_to_hidden_constraint,
)
from demonstrations.sudoku.constraints.lockout_line import lockout_line_constraint
from demonstrations.sudoku.constraints.modular_line import modular_line_constraints, modular_window_constraint
from demonstrations.sudoku.constraints.palindrome_line import (
    equal_pair_constraint,
    palindrome_line_constraint,
    palindrome_pair_constraints,
)
from demonstrations.sudoku.constraints.parity_cell import (
    even_cell_constraint,
    odd_cell_constraint,
    parity_cell_constraint,
)
from demonstrations.sudoku.constraints.quadruple import quadruple_constraint
from demonstrations.sudoku.constraints.red_dot import red_dot_constraint
from demonstrations.sudoku.constraints.red_line import red_line_constraint, red_line_constraint_from_odd
from demonstrations.sudoku.constraints.region_sum_line import (
    region_sum_line_constraints,
    region_sum_segment_constraint,
    same_region_sum_constraint,
)
from demonstrations.sudoku.constraints.renban_line import renban_line_constraint, renban_line_constraints
from demonstrations.sudoku.constraints.sequence_line import sequence_line_constraint
from demonstrations.sudoku.constraints.thermometer_line import (
    slow_thermometer_line_constraints,
    thermometer_line_constraints,
    thermometer_pair_constraint,
)
from demonstrations.sudoku.constraints.whisper_line import (
    dutch_whisper_constraint,
    german_whisper_constraint,
    whisper_line_constraints,
    whisper_pair_constraint,
)
from demonstrations.sudoku.constraints.white_dot import white_dot_constraint
from demonstrations.sudoku.constraints.xv import (
    sum_pair_constraint,
    v_constraint,
    v_marker_constraint,
    x_constraint,
    x_marker_constraint,
    xv_constraints,
)
from demonstrations.sudoku.constraints.zipper_line import (
    pair_sum_to_digit_constraint,
    zipper_line_constraints,
)
