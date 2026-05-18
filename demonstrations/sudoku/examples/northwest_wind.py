from demonstrations.sudoku.examples.challenge100_helpers import (
    get_assignment_as_constraint_network_from_cores,
    get_variant_constraint_cores,
)


PUZZLE_INDEX = 63
SUDOKU_NUM = 3
INITIAL_BOARD = (
    "................................................................................."
)
CAGES = ()
NO_REPEAT_CAGES = ()
ARROWS = (
    ("r2c1", ("r3c2", "r4c3", "r5c4")),
    ("r2c3", ("r3c4", "r4c5", "r5c6")),
    ("r2c5", ("r3c6", "r4c7", "r5c8")),
    ("r1c6", ("r2c7",)),
    ("r3c8", ("r4c9",)),
    ("r6c1", ("r7c2", "r8c3", "r9c4")),
    ("r6c3", ("r7c4", "r8c5", "r9c6")),
    ("r6c5", ("r7c6", "r8c7", "r9c8")),
    ("r6c7", ("r7c8",)),
    ("r7c9", ("r8c8", "r9c9")),
)
EDGE_MARKERS = ()
LINES = ()
QUADRUPLES = ()
PARITY_CELLS = (("even", "r6c9"),)
LITTLE_KILLERS = ()
INEQUALITIES = ()
ANTI_KNIGHT = False


def _check_num(num):
    if num != SUDOKU_NUM:
        raise ValueError(
            f"Puzzle {PUZZLE_INDEX} is represented with num={SUDOKU_NUM}, not num={num}."
        )


def get_northwest_wind_constraint_cores(num=SUDOKU_NUM, coreType=None):
    _check_num(num)
    return get_variant_constraint_cores(
        "northwest_wind",
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
        constraint_cores=get_northwest_wind_constraint_cores(
            num=num, coreType=coreType
        ),
        startAssignment=startAssignment,
        initial_board=INITIAL_BOARD,
        include_initial_board=include_initial_board,
    )
