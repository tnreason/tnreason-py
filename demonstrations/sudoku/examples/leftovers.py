from demonstrations.sudoku.examples.challenge100_helpers import (
    get_assignment_as_constraint_network_from_cores,
    get_variant_constraint_cores,
)


PUZZLE_INDEX = 97
SUDOKU_NUM = 3
INITIAL_BOARD = (
    "................................................................................."
)
CAGES = (
    (("r8c1", "r8c2"), 15),
    (("r1c8", "r2c8"), 15),
    (("r1c4", "r1c5", "r1c6"), 15),
    (("r4c1", "r5c1", "r6c1"), 15),
)
NO_REPEAT_CAGES = (
    ("r4c8", "r5c8", "r6c8", "r7c8", "r8c4", "r8c5", "r8c6", "r8c7", "r8c8"),
)
ARROWS = (
    ("r9c9", ("r8c9", "r7c9", "r6c9")),
    ("r9c9", ("r9c8", "r9c7", "r9c6")),
    ("r8c3", ("r7c3", "r6c3", "r5c2")),
    ("r3c8", ("r3c7", "r3c6", "r2c5")),
    ("r1c1", ("r2c2", "r3c3", "r4c4")),
    ("r7c7", ("r6c6", "r5c5", "r4c4")),
    ("r1c9", ("r2c9", "r3c9")),
)
EDGE_MARKERS = (("white_dot", "r3c2", "r4c2"), ("white_dot", "r9c4", "r9c5"))
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


def get_leftovers_constraint_cores(num=SUDOKU_NUM, coreType=None):
    _check_num(num)
    return get_variant_constraint_cores(
        "leftovers",
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
        constraint_cores=get_leftovers_constraint_cores(num=num, coreType=coreType),
        startAssignment=startAssignment,
        initial_board=INITIAL_BOARD,
        include_initial_board=include_initial_board,
    )
