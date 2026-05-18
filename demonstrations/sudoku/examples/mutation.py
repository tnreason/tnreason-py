from demonstrations.sudoku.examples.challenge100_helpers import (
    get_assignment_as_constraint_network_from_cores,
    get_variant_constraint_cores,
)


PUZZLE_INDEX = 37
SUDOKU_NUM = 3
INITIAL_BOARD = (
    "................................................................................."
)
CAGES = ()
NO_REPEAT_CAGES = ()
ARROWS = ()
EDGE_MARKERS = ()
LINES = (
    (
        "whisper",
        (
            "r2c2",
            "r3c3",
            "r4c4",
            "r3c5",
            "r2c6",
            "r2c7",
            "r2c8",
            "r3c7",
            "r4c6",
            "r5c7",
            "r6c8",
            "r7c8",
        ),
    ),
    (
        "whisper",
        (
            "r8c8",
            "r7c7",
            "r6c6",
            "r7c5",
            "r8c4",
            "r8c3",
            "r8c2",
            "r7c3",
            "r6c4",
            "r5c3",
            "r4c2",
            "r3c2",
        ),
    ),
)
QUADRUPLES = (
    (("r2c2", "r2c3", "r3c2", "r3c3"), (1, 2, 3, 8)),
    (("r2c7", "r2c8", "r3c7", "r3c8"), (2, 5)),
    (("r7c2", "r7c3", "r8c2", "r8c3"), (1, 7)),
    (("r5c7", "r5c8", "r6c7", "r6c8"), (4, 6, 9)),
    (("r4c2", "r4c3", "r5c2", "r5c3"), (3,)),
    (("r7c4", "r7c5", "r8c4", "r8c5"), (3, 8)),
    (("r7c7", "r7c8", "r8c7", "r8c8"), (2, 3, 6, 7)),
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


def get_mutation_constraint_cores(num=SUDOKU_NUM, coreType=None):
    _check_num(num)
    return get_variant_constraint_cores(
        "mutation",
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
        constraint_cores=get_mutation_constraint_cores(num=num, coreType=coreType),
        startAssignment=startAssignment,
        initial_board=INITIAL_BOARD,
        include_initial_board=include_initial_board,
    )
