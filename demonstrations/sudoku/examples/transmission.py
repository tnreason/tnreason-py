from demonstrations.sudoku.examples.challenge100_helpers import (
    get_assignment_as_constraint_network_from_cores,
    get_variant_constraint_cores,
)


PUZZLE_INDEX = 57
SUDOKU_NUM = 3
INITIAL_BOARD = (
    "................................................................................."
)
CAGES = (
    (("r3c1", "r4c1", "r4c2"), 11),
    (("r7c7", "r7c8", "r8c7"), 15),
    (("r8c6", "r9c6", "r9c7"), 17),
    (("r8c8", "r8c9", "r9c8"), 18),
    (("r6c8", "r6c9", "r7c9"), 16),
    (("r5c1", "r5c2"), 9),
    (("r1c3", "r1c4", "r2c4"), 15),
    (("r2c3", "r3c2", "r3c3"), 14),
    (("r1c2", "r2c1", "r2c2"), 16),
    (("r8c5", "r9c5"), 13),
    (("r5c4", "r6c4", "r6c5"), 21),
    (("r6c3", "r7c3", "r7c4"), 9),
    (("r3c6", "r3c7", "r4c7"), 20),
    (("r4c5", "r4c6", "r5c6"), 13),
)
NO_REPEAT_CAGES = ()
ARROWS = ()
EDGE_MARKERS = ()
LINES = (
    (
        "modular",
        ("r5c1", "r4c1", "r3c1", "r2c1", "r1c1", "r1c2", "r1c3", "r1c4", "r1c5"),
    ),
    ("modular", ("r5c2", "r4c2", "r3c2", "r2c2", "r2c3", "r2c4", "r2c5")),
    (
        "modular",
        ("r9c5", "r9c6", "r9c7", "r9c8", "r9c9", "r8c9", "r7c9", "r6c9", "r5c9"),
    ),
    ("modular", ("r8c5", "r8c6", "r8c7", "r8c8", "r7c8", "r6c8", "r5c8")),
    ("modular", ("r3c6", "r4c6", "r4c7")),
    ("modular", ("r6c3", "r6c4", "r7c4")),
    ("modular", ("r4c5", "r4c4", "r5c4")),
    ("modular", ("r6c5", "r6c6", "r5c6")),
    ("modular", ("r7c3", "r7c2", "r8c2")),
    ("modular", ("r3c7", "r3c8", "r2c8")),
)
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


def get_transmission_constraint_cores(num=SUDOKU_NUM, coreType=None):
    _check_num(num)
    return get_variant_constraint_cores(
        "transmission",
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
        constraint_cores=get_transmission_constraint_cores(num=num, coreType=coreType),
        startAssignment=startAssignment,
        initial_board=INITIAL_BOARD,
        include_initial_board=include_initial_board,
    )
