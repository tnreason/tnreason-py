from demonstrations.sudoku.examples.challenge100_helpers import (
    get_assignment_as_constraint_network_from_cores,
    get_variant_constraint_cores,
)


PUZZLE_INDEX = 64
SUDOKU_NUM = 3
INITIAL_BOARD = (
    "................................................................................."
)
CAGES = (
    (("r8c1", "r9c1", "r9c2"), 22),
    (("r5c2", "r6c2", "r7c2"), 22),
    (("r8c3", "r8c4", "r8c5"), 22),
    (("r1c8", "r1c9", "r2c9"), 8),
    (("r2c5", "r2c6", "r2c7"), 8),
    (("r3c8", "r4c8", "r5c8"), 8),
    (("r6c3", "r6c4", "r7c4"), 20),
    (("r3c6", "r4c6", "r4c7"), 20),
    (("r1c3", "r1c4"), 10),
    (("r9c6", "r9c7"), 10),
)
NO_REPEAT_CAGES = (("r2c1", "r2c2", "r3c1", "r4c1"), ("r6c9", "r7c9", "r8c8", "r8c9"))
ARROWS = ()
EDGE_MARKERS = ()
LINES = ()
QUADRUPLES = ()
PARITY_CELLS = (
    ("odd", "r4c1"),
    ("odd", "r2c2"),
    ("odd", "r1c6"),
    ("odd", "r9c4"),
    ("odd", "r8c8"),
    ("odd", "r6c9"),
    ("even", "r6c4"),
    ("even", "r5c4"),
    ("even", "r4c6"),
    ("even", "r5c6"),
)
LITTLE_KILLERS = ()
INEQUALITIES = ()
ANTI_KNIGHT = False


def _check_num(num):
    if num != SUDOKU_NUM:
        raise ValueError(
            f"Puzzle {PUZZLE_INDEX} is represented with num={SUDOKU_NUM}, not num={num}."
        )


def get_uraninite_constraint_cores(num=SUDOKU_NUM, coreType=None):
    _check_num(num)
    return get_variant_constraint_cores(
        "uraninite",
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
        constraint_cores=get_uraninite_constraint_cores(num=num, coreType=coreType),
        startAssignment=startAssignment,
        initial_board=INITIAL_BOARD,
        include_initial_board=include_initial_board,
    )
