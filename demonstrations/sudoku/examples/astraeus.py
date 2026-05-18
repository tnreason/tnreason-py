from demonstrations.sudoku.examples.challenge100_helpers import (
    get_assignment_as_constraint_network_from_cores,
    get_variant_constraint_cores,
)


PUZZLE_INDEX = 28
SUDOKU_NUM = 3
INITIAL_BOARD = (
    "................................................................................."
)
CAGES = ()
NO_REPEAT_CAGES = ()
ARROWS = ()
EDGE_MARKERS = (("v", "r1c1", "r2c1"),)
LINES = (
    ("thermo", ("r5c3", "r4c4", "r4c5", "r5c6")),
    ("thermo", ("r5c4", "r6c5", "r6c6", "r5c7")),
    ("thermo", ("r3c9", "r2c9", "r1c9")),
    ("thermo", ("r7c1", "r8c1", "r9c1")),
    ("renban", ("r8c4", "r7c4", "r7c5")),
    ("renban", ("r8c5", "r9c5", "r9c6")),
    ("renban", ("r7c6", "r8c6")),
    ("renban", ("r6c9", "r6c8", "r7c8", "r8c8", "r9c8")),
    ("renban", ("r4c9", "r4c8", "r3c8", "r2c8", "r1c8")),
    ("renban", ("r4c1", "r4c2", "r3c2", "r2c2", "r1c2")),
    ("renban", ("r6c1", "r6c2", "r7c2", "r8c2", "r9c2")),
    ("renban", ("r6c3", "r7c3")),
    ("renban", ("r2c4", "r3c4", "r3c5")),
    ("renban", ("r1c4", "r1c5", "r2c5")),
    ("renban", ("r2c6", "r3c6")),
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


def get_astraeus_constraint_cores(num=SUDOKU_NUM, coreType=None):
    _check_num(num)
    return get_variant_constraint_cores(
        "astraeus",
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
        constraint_cores=get_astraeus_constraint_cores(num=num, coreType=coreType),
        startAssignment=startAssignment,
        initial_board=INITIAL_BOARD,
        include_initial_board=include_initial_board,
    )
