from demonstrations.sudoku.examples.challenge100_helpers import (
    get_assignment_as_constraint_network_from_cores,
    get_variant_constraint_cores,
)


PUZZLE_INDEX = 23
SUDOKU_NUM = 3
INITIAL_BOARD = (
    "................................................................................."
)
CAGES = ()
NO_REPEAT_CAGES = ()
ARROWS = ()
EDGE_MARKERS = (
    ("white_dot", "r4c8", "r4c9"),
    ("white_dot", "r5c8", "r5c9"),
    ("white_dot", "r6c8", "r6c9"),
    ("white_dot", "r5c1", "r5c2"),
    ("white_dot", "r8c3", "r9c3"),
    ("white_dot", "r7c1", "r8c1"),
    ("white_dot", "r1c1", "r2c1"),
    ("white_dot", "r7c7", "r7c8"),
    ("white_dot", "r7c1", "r7c2"),
    ("white_dot", "r9c8", "r9c9"),
    ("white_dot", "r8c5", "r8c6"),
    ("white_dot", "r1c4", "r2c4"),
    ("white_dot", "r7c6", "r8c6"),
    ("white_dot", "r2c7", "r3c7"),
    ("white_dot", "r1c2", "r1c3"),
    ("white_dot", "r1c5", "r2c5"),
    ("black_dot", "r3c2", "r4c2"),
    ("black_dot", "r4c7", "r4c8"),
    ("black_dot", "r2c3", "r3c3"),
    ("black_dot", "r9c2", "r9c3"),
    ("black_dot", "r8c8", "r9c8"),
    ("red_line", "r3c2", "r3c3"),
    ("red_line", "r3c3", "r3c4"),
    ("red_line", "r3c4", "r3c5"),
    ("red_line", "r3c5", "r3c6"),
    ("red_line", "r3c6", "r4c7"),
    ("red_line", "r4c7", "r5c8"),
    ("red_line", "r5c8", "r6c7"),
    ("red_line", "r6c7", "r7c6"),
    ("red_line", "r7c6", "r7c5"),
    ("red_line", "r7c5", "r7c4"),
    ("red_line", "r7c4", "r7c3"),
    ("red_line", "r7c3", "r7c2"),
    ("red_line", "r4c1", "r4c2"),
    ("red_line", "r4c2", "r5c3"),
    ("red_line", "r5c3", "r6c4"),
    ("red_line", "r6c4", "r7c4"),
    ("red_line", "r6c1", "r6c2"),
    ("red_line", "r6c2", "r5c3"),
    ("red_line", "r5c3", "r4c4"),
)
LINES = ()
QUADRUPLES = ()
PARITY_CELLS = ()
LITTLE_KILLERS = ()
INEQUALITIES = ()
ANTI_KNIGHT = False


def _check_num(num):
    if num != SUDOKU_NUM:
        raise ValueError(
            f"Puzzle {PUZZLE_INDEX} is represented with num={SUDOKU_NUM}, not num={num}."
        )


def get_parity_fish_constraint_cores(num=SUDOKU_NUM, coreType=None):
    _check_num(num)
    return get_variant_constraint_cores(
        "parity_fish",
        num=num,
        cages=CAGES,
        no_repeat_cages=NO_REPEAT_CAGES,
        arrows=ARROWS,
        edge_markers=EDGE_MARKERS,
        lines=LINES,
        quadruples=QUADRUPLES,
        parity_cells=PARITY_CELLS,
        little_killers=LITTLE_KILLERS,
        inequalities=INEQUALITIES,
        anti_knight=ANTI_KNIGHT,
        coreType=coreType,
    )


def get_assignment_as_constraint_network(
    num=SUDOKU_NUM,
    startAssignment=None,
    include_initial_board=True,
    coreType=None,
):
    _check_num(num)
    return get_assignment_as_constraint_network_from_cores(
        num=num,
        constraint_cores=get_parity_fish_constraint_cores(num=num, coreType=coreType),
        startAssignment=startAssignment,
        initial_board=INITIAL_BOARD,
        include_initial_board=include_initial_board,
    )
