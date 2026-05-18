from demonstrations.sudoku.examples.challenge100_helpers import (
    get_assignment_as_constraint_network_from_cores,
    get_variant_constraint_cores,
)


PUZZLE_INDEX = 49
SUDOKU_NUM = 3
INITIAL_BOARD = (
    "4.......8...............................................................1.......3"
)
CAGES = ()
NO_REPEAT_CAGES = ()
ARROWS = ()
EDGE_MARKERS = ()
LINES = (
    ("slow_thermo", ("r4c1", "r3c2", "r3c3", "r4c4")),
    ("slow_thermo", ("r4c6", "r3c7", "r3c8", "r4c9")),
    ("slow_thermo", ("r7c5", "r8c5", "r9c4")),
    ("slow_thermo", ("r7c5", "r8c5", "r9c6")),
    ("whisper", ("r2c1", "r3c2", "r3c3", "r2c4")),
    ("whisper", ("r2c6", "r3c7", "r3c8", "r2c9")),
    ("whisper", ("r4c2", "r5c2")),
    ("whisper", ("r4c3", "r5c3")),
    ("whisper", ("r4c7", "r5c7")),
    ("whisper", ("r4c8", "r5c8")),
    ("whisper", ("r6c3", "r7c4", "r7c5", "r7c6", "r6c7")),
    ("whisper", ("r7c5", "r7c4", "r7c3", "r8c2")),
    ("whisper", ("r7c5", "r7c6", "r7c7", "r8c8")),
    ("renban", ("r4c1", "r5c1", "r6c2", "r6c3", "r5c4", "r4c4")),
    ("renban", ("r4c6", "r5c6", "r6c7", "r6c8", "r5c9", "r4c9")),
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


def get_the_two_headed_martian_constraint_cores(num=SUDOKU_NUM, coreType=None):
    _check_num(num)
    return get_variant_constraint_cores(
        "the_two_headed_martian",
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
        constraint_cores=get_the_two_headed_martian_constraint_cores(
            num=num, coreType=coreType
        ),
        startAssignment=startAssignment,
        initial_board=INITIAL_BOARD,
        include_initial_board=include_initial_board,
    )
