from demonstrations.sudoku.examples.challenge100_helpers import (
    get_assignment_as_constraint_network_from_cores,
    get_variant_constraint_cores,
)


PUZZLE_INDEX = 42
SUDOKU_NUM = 3
INITIAL_BOARD = (
    "................................................................................."
)
CAGES = ()
NO_REPEAT_CAGES = ()
ARROWS = ()
EDGE_MARKERS = ()
LINES = (
    ("fibonacci", ("r9c7", "r8c7", "r7c7", "r7c8", "r7c9")),
    ("fibonacci", ("r1c3", "r2c3", "r3c3", "r3c2", "r3c1")),
    ("fibonacci", ("r5c3", "r5c4", "r5c5", "r5c6", "r5c7")),
    ("fibonacci", ("r9c1", "r8c1", "r8c2", "r9c2")),
    ("fibonacci", ("r1c8", "r2c8", "r2c9", "r1c9")),
    ("fibonacci", ("r4c2", "r5c2", "r6c2")),
    ("fibonacci", ("r4c8", "r5c8", "r6c8")),
    ("fibonacci", ("r8c4", "r8c5", "r8c6")),
    ("fibonacci", ("r2c4", "r2c5", "r2c6")),
    ("fibonacci", ("r6c1", "r7c1", "r7c2")),
    ("fibonacci", ("r6c3", "r6c4", "r7c4")),
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


def get_fibonacci_lines_constraint_cores(num=SUDOKU_NUM, coreType=None):
    _check_num(num)
    return get_variant_constraint_cores(
        "fibonacci_lines",
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
        constraint_cores=get_fibonacci_lines_constraint_cores(
            num=num, coreType=coreType
        ),
        startAssignment=startAssignment,
        initial_board=INITIAL_BOARD,
        include_initial_board=include_initial_board,
    )
