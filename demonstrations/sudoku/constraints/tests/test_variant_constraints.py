from demonstrations.sudoku.constraints import (
    all_anti_king_constraints,
    all_anti_knight_constraints,
    arrow_sum_constraint,
    balanced_four_cell_line_constraint,
    diagonal_product_constraint,
    entropic_window_constraint,
    even_cell_constraint,
    fibonacci_line_constraint,
    gray_dot_constraint,
    inequality_constraint,
    killer_cage_constraint,
    little_killer_constraint,
    lockout_line_constraint,
    modular_window_constraint,
    odd_cell_constraint,
    palindrome_pair_constraints,
    quadruple_constraint,
    red_dot_constraint,
    region_sum_line_constraints,
    renban_line_constraint,
    sequence_line_constraint,
    slow_thermometer_line_constraints,
    thermometer_pair_constraint,
    v_constraint,
    whisper_pair_constraint,
    x_constraint,
    zipper_line_constraints,
)


def test_pair_rules():
    assert v_constraint("a", "b")[{"a": 0, "b": 3}] == 1
    assert v_constraint("a", "b")[{"a": 0, "b": 4}] == 0
    assert x_constraint("a", "b")[{"a": 0, "b": 8}] == 1
    assert whisper_pair_constraint("a", "b")[{"a": 0, "b": 5}] == 1
    assert whisper_pair_constraint("a", "b")[{"a": 0, "b": 4}] == 0
    assert inequality_constraint("a", "b", "<")[{"a": 1, "b": 2}] == 1
    assert inequality_constraint("a", "b", "<")[{"a": 2, "b": 1}] == 0
    assert red_dot_constraint("a", "b")[{"a": 0, "b": 3}] == 1
    assert red_dot_constraint("a", "b")[{"a": 0, "b": 1}] == 0
    assert gray_dot_constraint("a", "b")[{"a": 0, "b": 1}] == 1
    assert gray_dot_constraint("a", "b")[{"a": 1, "b": 3}] == 1
    assert gray_dot_constraint("a", "b")[{"a": 1, "b": 4}] == 0


def test_cell_and_chess_rules():
    assert odd_cell_constraint("a")[{"a": 0}] == 1
    assert odd_cell_constraint("a")[{"a": 1}] == 0
    assert even_cell_constraint("a")[{"a": 1}] == 1
    assert len(all_anti_knight_constraints(row_count=4, col_count=4, sudokuNum=2)) == 24
    assert len(all_anti_king_constraints(row_count=2, col_count=2, sudokuNum=2)) == 6


def test_sum_and_set_rules():
    assert killer_cage_constraint(["a", "b"], 5)[{"a": 0, "b": 3}] == 1
    assert killer_cage_constraint(["a", "b"], 5)[{"a": 1, "b": 1}] == 0
    assert arrow_sum_constraint("c", ["a", "b"])[{"c": 2, "a": 0, "b": 1}] == 1
    assert arrow_sum_constraint("c", ["a", "b"])[{"c": 0, "a": 0, "b": 1}] == 0
    assert little_killer_constraint(["a", "b", "c"], 6)[{"a": 0, "b": 1, "c": 2}] == 1
    assert quadruple_constraint(["a", "b", "c", "d"], [1, 4], sudokuNum=2)[
        {"a": 0, "b": 1, "c": 2, "d": 3}
    ] == 1
    assert quadruple_constraint(["a", "b", "c", "d"], [1, 4], sudokuNum=2)[
        {"a": 0, "b": 1, "c": 2, "d": 2}
    ] == 0


def test_line_rules():
    assert renban_line_constraint(["a", "b", "c"])[{"a": 2, "b": 0, "c": 1}] == 1
    assert renban_line_constraint(["a", "b", "c"])[{"a": 0, "b": 1, "c": 3}] == 0
    assert entropic_window_constraint("a", "b", "c")[{"a": 0, "b": 3, "c": 6}] == 1
    assert entropic_window_constraint("a", "b", "c")[{"a": 0, "b": 1, "c": 6}] == 0
    assert modular_window_constraint("a", "b", "c")[{"a": 0, "b": 1, "c": 2}] == 1
    assert modular_window_constraint("a", "b", "c")[{"a": 0, "b": 3, "c": 6}] == 0
    assert balanced_four_cell_line_constraint(["a", "b", "c", "d"], sudokuNum=2)[
        {"a": 0, "b": 3, "c": 1, "d": 2}
    ] == 1
    assert fibonacci_line_constraint(["a", "b", "c", "d"])[{"a": 0, "b": 1, "c": 2, "d": 4}] == 1
    assert sequence_line_constraint(["a", "b", "c"])[{"a": 0, "b": 2, "c": 4}] == 1
    assert lockout_line_constraint(["a", "b", "c"])[{"a": 2, "b": 0, "c": 7}] == 1
    assert lockout_line_constraint(["a", "b", "c"])[{"a": 2, "b": 4, "c": 7}] == 0


def test_decomposed_line_rules():
    assert thermometer_pair_constraint("a", "b")[{"a": 0, "b": 1}] == 1
    assert thermometer_pair_constraint("a", "b")[{"a": 1, "b": 1}] == 0
    slow = slow_thermometer_line_constraints(["a", "b", "c"])
    assert slow["slow_thermo_0"][{"a": 1, "b": 1}] == 1
    pal = palindrome_pair_constraints(["a", "b", "c", "d"])
    assert pal["palindrome_0"][{"a": 2, "d": 2}] == 1
    zipper = zipper_line_constraints(["a", "m", "b"])
    assert zipper["zipper_0"][{"m": 2, "a": 0, "b": 1}] == 1
    assert zipper["zipper_0"][{"m": 1, "a": 0, "b": 1}] == 0


def test_hidden_sum_and_product_rules():
    region = region_sum_line_constraints([["a", "b"], ["c"]], sum_var="s", sudokuNum=2, max_sum=8)
    assert region["region_sum_0"][{"s": 3, "a": 0, "b": 1}] == 1
    assert region["region_sum_1"][{"s": 3, "c": 2}] == 1
    assert region["region_sum_1"][{"s": 3, "c": 3}] == 0
    assert diagonal_product_constraint(["a", "b"], 6)[{"a": 1, "b": 2}] == 1
    assert diagonal_product_constraint(["a", "b"], 6)[{"a": 0, "b": 2}] == 0
