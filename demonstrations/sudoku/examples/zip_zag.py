from demonstrations.sudoku.examples.challenge100_helpers import (
    get_assignment_as_constraint_network_from_cores,
    get_variant_constraint_cores,
)


PUZZLE_INDEX = 31
SUDOKU_NUM = 3
INITIAL_BOARD = (
    "................................................................................."
)
CAGES = ()
NO_REPEAT_CAGES = ()
ARROWS = ()
EDGE_MARKERS = ()
LINES = (
    ("zipper", ("r1c9", "r2c8", "r3c7", "r3c6", "r3c5", "r2c5", "r1c5")),
    ("zipper", ("r5c9", "r5c8", "r5c7", "r5c6", "r6c7", "r6c8", "r7c9")),
    ("zipper", ("r8c9", "r7c8", "r7c7", "r7c6", "r8c7", "r8c8", "r9c9")),
    ("zipper", ("r9c5", "r8c5", "r7c5", "r6c4", "r5c3", "r5c2", "r5c1")),
    ("zipper", ("r1c4", "r2c4", "r3c4", "r4c4", "r4c3", "r3c3", "r3c2")),
    ("zipper", ("r9c2", "r8c2", "r7c3", "r6c3", "r6c2", "r7c1", "r8c1")),
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


def get_zip_zag_constraint_cores(num=SUDOKU_NUM, coreType=None):
    _check_num(num)
    return get_variant_constraint_cores(
        "zip_zag",
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
        constraint_cores=get_zip_zag_constraint_cores(num=num, coreType=coreType),
        startAssignment=startAssignment,
        initial_board=INITIAL_BOARD,
        include_initial_board=include_initial_board,
    )
