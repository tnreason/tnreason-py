from demonstrations.sudoku.examples.challenge100_helpers import (
    get_assignment_as_constraint_network_from_cores,
    get_variant_constraint_cores,
)


PUZZLE_INDEX = 54
SUDOKU_NUM = 3
INITIAL_BOARD = (
    "................................................................................."
)
CAGES = (
    (("r9c2", "r9c3", "r9c4", "r9c5", "r9c6"), 19),
    (("r4c1", "r5c1", "r6c1", "r7c1", "r8c1"), 16),
    (("r2c7", "r2c8", "r3c7", "r3c8"), 12),
)
NO_REPEAT_CAGES = ()
ARROWS = (
    ("r2c4", ("r1c3", "r2c2")),
    ("r3c4", ("r3c3", "r3c2")),
    ("r1c4", ("r2c3", "r1c2")),
    ("r6c8", ("r7c7", "r8c8")),
    ("r6c7", ("r7c8", "r8c7")),
    ("r6c9", ("r7c9", "r8c9")),
    ("r5c7", ("r5c8", "r4c8")),
    ("r3c5", ("r2c5", "r2c6")),
    ("r3c6", ("r4c5", "r5c4")),
    ("r4c7", ("r5c6", "r6c5")),
)
EDGE_MARKERS = (
    ("white_dot", "r6c7", "r6c8"),
    ("white_dot", "r6c8", "r6c9"),
    ("white_dot", "r1c4", "r2c4"),
    ("white_dot", "r2c4", "r3c4"),
    ("white_dot", "r3c5", "r3c6"),
    ("white_dot", "r4c7", "r5c7"),
    ("white_dot", "r6c1", "r7c1"),
    ("white_dot", "r9c3", "r9c4"),
    ("white_dot", "r4c3", "r5c3"),
    ("white_dot", "r7c5", "r7c6"),
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


def get_what_s_the_difference_constraint_cores(num=SUDOKU_NUM, coreType=None):
    _check_num(num)
    return get_variant_constraint_cores(
        "what_s_the_difference",
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
        constraint_cores=get_what_s_the_difference_constraint_cores(
            num=num, coreType=coreType
        ),
        startAssignment=startAssignment,
        initial_board=INITIAL_BOARD,
        include_initial_board=include_initial_board,
    )
