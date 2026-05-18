from demonstrations.sudoku.examples.challenge100_helpers import (
    get_assignment_as_constraint_network_from_cores,
    get_variant_constraint_cores,
)


PUZZLE_INDEX = 34
SUDOKU_NUM = 3
INITIAL_BOARD = (
    "................................................................................."
)
CAGES = (
    (("r1c1", "r1c2", "r2c1", "r2c2"), 17),
    (("r1c5", "r1c6", "r2c5", "r2c6"), 11),
    (("r1c8", "r1c9", "r2c8", "r2c9"), 17),
    (("r8c1", "r8c2", "r9c1", "r9c2"), 20),
    (("r8c5", "r8c6", "r9c5", "r9c6"), 10),
    (("r8c8", "r8c9", "r9c8", "r9c9"), 15),
)
NO_REPEAT_CAGES = ()
ARROWS = (
    ("r1c1", ("r2c2", "r3c2", "r4c2")),
    ("r9c2", ("r8c1", "r7c1", "r6c1")),
    ("r2c8", ("r2c9", "r3c9", "r4c9")),
    ("r8c9", ("r8c8", "r7c8", "r6c8")),
    ("r5c2", ("r4c3", "r5c3", "r6c3")),
    ("r5c5", ("r4c4", "r5c4", "r6c4")),
    ("r5c8", ("r4c7", "r5c7", "r6c7")),
)
EDGE_MARKERS = ()
LINES = ()
QUADRUPLES = ()
PARITY_CELLS = (("odd", "r4c4"),)
LITTLE_KILLERS = ()
INEQUALITIES = ()
ANTI_KNIGHT = False


def _check_num(num):
    if num != SUDOKU_NUM:
        raise ValueError(
            f"Puzzle {PUZZLE_INDEX} is represented with num={SUDOKU_NUM}, not num={num}."
        )


def get_four_corner_shots_constraint_cores(num=SUDOKU_NUM, coreType=None):
    _check_num(num)
    return get_variant_constraint_cores(
        "four_corner_shots",
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
        constraint_cores=get_four_corner_shots_constraint_cores(
            num=num, coreType=coreType
        ),
        startAssignment=startAssignment,
        initial_board=INITIAL_BOARD,
        include_initial_board=include_initial_board,
    )
