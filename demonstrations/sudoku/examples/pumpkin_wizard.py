from demonstrations.sudoku.examples.challenge100_helpers import (
    get_assignment_as_constraint_network_from_cores,
    get_variant_constraint_cores,
)


PUZZLE_INDEX = 65
SUDOKU_NUM = 3
INITIAL_BOARD = (
    "................................................................................."
)
CAGES = ((("r1c4", "r1c5", "r1c6"), 11), (("r8c9", "r9c8", "r9c9"), 13))
NO_REPEAT_CAGES = ()
ARROWS = (("r2c4", ("r2c5", "r2c6")),)
EDGE_MARKERS = (
    ("gray_dot", "r2c6", "r3c6"),
    ("gray_dot", "r1c8", "r1c9"),
    ("gray_dot", "r9c8", "r9c9"),
)
LINES = (
    ("sequence", ("r5c4", "r5c3", "r4c3")),
    ("sequence", ("r3c1", "r2c2", "r1c2")),
    ("dutch_whisper", ("r4c3", "r5c4")),
    ("dutch_whisper", ("r6c3", "r7c4", "r6c5", "r7c6", "r7c7", "r6c7")),
    (
        "dutch_whisper",
        (
            "r8c6",
            "r8c7",
            "r7c8",
            "r6c8",
            "r5c8",
            "r4c8",
            "r3c7",
            "r3c6",
            "r3c5",
            "r4c4",
            "r3c3",
            "r4c2",
            "r5c2",
            "r6c2",
            "r7c2",
            "r8c3",
            "r8c4",
            "r7c5",
            "r8c6",
        ),
    ),
    ("dutch_whisper", ("r4c6", "r5c5", "r5c6", "r4c6")),
    ("lockout", ("r3c6", "r2c5", "r1c5", "r2c4", "r3c3")),
    ("lockout", ("r9c3", "r9c2", "r8c1")),
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


def get_pumpkin_wizard_constraint_cores(num=SUDOKU_NUM, coreType=None):
    _check_num(num)
    return get_variant_constraint_cores(
        "pumpkin_wizard",
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
        constraint_cores=get_pumpkin_wizard_constraint_cores(
            num=num, coreType=coreType
        ),
        startAssignment=startAssignment,
        initial_board=INITIAL_BOARD,
        include_initial_board=include_initial_board,
    )
