from demonstrations.sudoku.examples.challenge100_helpers import (
    get_assignment_as_constraint_network_from_cores,
    get_variant_constraint_cores,
)


PUZZLE_INDEX = 43
SUDOKU_NUM = 3
INITIAL_BOARD = (
    "................................................................................."
)
CAGES = (
    (("r4c5", "r5c5", "r6c5"), 15),
    (("r4c2", "r5c2", "r6c2"), 15),
    (("r4c8", "r5c8", "r6c8"), 15),
    (("r8c3", "r8c4", "r8c5", "r8c6", "r8c7"), 15),
    (("r2c3", "r2c4", "r2c5", "r2c6", "r2c7"), 15),
    (("r1c1", "r1c2", "r2c1"), 15),
    (("r1c8", "r1c9", "r2c9"), 15),
    (("r3c3", "r3c4"), 15),
    (("r3c6", "r3c7"), 15),
    (("r7c4", "r7c5", "r7c6"), 15),
)
NO_REPEAT_CAGES = ()
ARROWS = ()
EDGE_MARKERS = (
    ("black_dot", "r5c5", "r6c5"),
    ("black_dot", "r4c2", "r5c2"),
    ("black_dot", "r4c8", "r5c8"),
    ("black_dot", "r8c4", "r8c5"),
    ("black_dot", "r8c5", "r8c6"),
    ("white_dot", "r2c3", "r2c4"),
    ("white_dot", "r2c6", "r2c7"),
    ("white_dot", "r1c1", "r2c1"),
    ("white_dot", "r1c8", "r1c9"),
    ("white_dot", "r7c1", "r7c2"),
    ("white_dot", "r3c1", "r4c1"),
    ("white_dot", "r3c9", "r4c9"),
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


def get_challenge100_43_constraint_cores(num=SUDOKU_NUM, coreType=None):
    _check_num(num)
    return get_variant_constraint_cores(
        "challenge100_43",
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
        constraint_cores=get_challenge100_43_constraint_cores(
            num=num, coreType=coreType
        ),
        startAssignment=startAssignment,
        initial_board=INITIAL_BOARD,
        include_initial_board=include_initial_board,
    )
