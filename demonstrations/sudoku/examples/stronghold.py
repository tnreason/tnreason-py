from demonstrations.sudoku.examples.challenge100_helpers import (
    get_assignment_as_constraint_network_from_cores,
    get_variant_constraint_cores,
)


PUZZLE_INDEX = 67
SUDOKU_NUM = 3
INITIAL_BOARD = (
    "................................................................................."
)
CAGES = ()
NO_REPEAT_CAGES = ()
ARROWS = (
    ("r7c3", ("r7c4", "r7c5", "r6c5")),
    ("r7c7", ("r7c6", "r7c5")),
    ("r6c2", ("r5c2", "r6c1", "r5c1")),
    ("r6c8", ("r5c8", "r6c9", "r5c9")),
    ("r9c1", ("r8c1", "r9c2", "r8c2")),
    ("r9c9", ("r8c9", "r9c8", "r8c8")),
)
EDGE_MARKERS = ()
LINES = (
    ("region_sum", ("r4c5", "r4c6", "r4c7", "r3c7", "r2c7", "r1c7")),
    ("region_sum", ("r6c8", "r7c7")),
    ("region_sum", ("r6c2", "r7c3")),
    ("region_sum", ("r4c5", "r4c4", "r4c3", "r3c3", "r2c3", "r1c3")),
)
QUADRUPLES = ()
PARITY_CELLS = ()
LITTLE_KILLERS = ()
INEQUALITIES = (
    ("r1c4", ">", "r1c5"),
    ("r2c4", ">", "r2c5"),
    ("r2c5", "<", "r2c4"),
    ("r3c4", ">", "r3c5"),
    ("r4c4", "<", "r3c4"),
    ("r3c5", ">", "r4c5"),
    ("r4c5", "<", "r3c5"),
    ("r3c7", "<", "r3c6"),
    ("r4c6", "<", "r3c6"),
    ("r2c6", ">", "r2c7"),
    ("r2c7", "<", "r2c6"),
    ("r1c7", "<", "r1c6"),
    ("r2c5", "<", "r1c5"),
    ("r2c2", ">", "r2c3"),
    ("r2c3", "<", "r2c2"),
    ("r2c2", ">", "r3c2"),
    ("r3c2", "<", "r2c2"),
    ("r2c8", ">", "r2c9"),
    ("r2c9", "<", "r2c8"),
    ("r2c8", ">", "r3c8"),
    ("r3c8", "<", "r2c8"),
)
ANTI_KNIGHT = False


def _check_num(num):
    if num != SUDOKU_NUM:
        raise ValueError(
            f"Puzzle {PUZZLE_INDEX} is represented with num={SUDOKU_NUM}, not num={num}."
        )


def get_stronghold_constraint_cores(num=SUDOKU_NUM, coreType=None):
    _check_num(num)
    return get_variant_constraint_cores(
        "stronghold",
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
        constraint_cores=get_stronghold_constraint_cores(num=num, coreType=coreType),
        startAssignment=startAssignment,
        initial_board=INITIAL_BOARD,
        include_initial_board=include_initial_board,
    )
