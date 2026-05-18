from demonstrations.sudoku.examples.challenge100_helpers import (
    get_assignment_as_constraint_network_from_cores,
    get_variant_constraint_cores,
)


PUZZLE_INDEX = 48
SUDOKU_NUM = 3
INITIAL_BOARD = (
    "................................................................................."
)
CAGES = ()
NO_REPEAT_CAGES = ()
ARROWS = ()
EDGE_MARKERS = (("white_dot", "r8c7", "r8c8"),)
LINES = (
    ("whisper", ("r3c1", "r4c1", "r3c2", "r3c3", "r4c4")),
    ("palindrome", ("r3c1", "r4c1", "r3c2", "r3c3", "r4c4")),
    ("whisper", ("r1c4", "r2c4", "r3c4", "r3c5", "r4c5", "r5c5", "r6c5")),
    ("palindrome", ("r1c4", "r2c4", "r3c4", "r3c5", "r4c5", "r5c5", "r6c5")),
    ("whisper", ("r2c6", "r2c7", "r3c7", "r4c8", "r3c9")),
    ("palindrome", ("r2c6", "r2c7", "r3c7", "r4c8", "r3c9")),
    ("whisper", ("r5c7", "r6c7", "r7c7", "r7c6", "r8c6")),
    ("palindrome", ("r5c7", "r6c7", "r7c7", "r7c6", "r8c6")),
    ("whisper", ("r6c3", "r7c3", "r8c3", "r9c3", "r9c4", "r8c4", "r7c4")),
    ("palindrome", ("r6c3", "r7c3", "r8c3", "r9c3", "r9c4", "r8c4", "r7c4")),
)
QUADRUPLES = (
    (("r2c1", "r2c2", "r3c1", "r3c2"), (1, 3, 4, 8)),
    (("r2c8", "r2c9", "r3c8", "r3c9"), (2, 3, 4, 6)),
    (("r5c2", "r5c3", "r6c2", "r6c3"), (2, 3, 7, 8)),
    (("r5c7", "r5c8", "r6c7", "r6c8"), (2, 6, 7, 9)),
    (("r7c8", "r7c9", "r8c8", "r8c9"), (2, 5, 7, 8)),
    (("r7c1", "r7c2", "r8c1", "r8c2"), (3, 4, 6, 9)),
)
PARITY_CELLS = ()
LITTLE_KILLERS = ()
INEQUALITIES = ()
ANTI_KNIGHT = False


def _check_num(num):
    if num != SUDOKU_NUM:
        raise ValueError(
            f"Puzzle {PUZZLE_INDEX} is represented with num={SUDOKU_NUM}, not num={num}."
        )


def get_broken_paperclip_constraint_cores(num=SUDOKU_NUM, coreType=None):
    _check_num(num)
    return get_variant_constraint_cores(
        "broken_paperclip",
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
        constraint_cores=get_broken_paperclip_constraint_cores(
            num=num, coreType=coreType
        ),
        startAssignment=startAssignment,
        initial_board=INITIAL_BOARD,
        include_initial_board=include_initial_board,
    )
