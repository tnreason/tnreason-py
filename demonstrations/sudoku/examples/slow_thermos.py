from demonstrations.sudoku.examples.challenge100_helpers import (
    get_assignment_as_constraint_network_from_cores,
    get_variant_constraint_cores,
)


PUZZLE_INDEX = 33
SUDOKU_NUM = 3
INITIAL_BOARD = (
    "....5.......................................................................1...."
)
CAGES = ()
NO_REPEAT_CAGES = ()
ARROWS = ()
EDGE_MARKERS = ()
LINES = (
    (
        "slow_thermo",
        (
            "r4c2",
            "r5c2",
            "r6c2",
            "r7c3",
            "r8c4",
            "r8c5",
            "r8c6",
            "r7c7",
            "r6c8",
            "r5c8",
        ),
    ),
    (
        "slow_thermo",
        (
            "r2c3",
            "r3c4",
            "r4c5",
            "r3c6",
            "r2c7",
            "r3c8",
            "r4c7",
            "r5c6",
            "r6c7",
            "r7c8",
            "r8c7",
            "r7c6",
            "r6c5",
            "r7c4",
            "r8c3",
            "r7c2",
            "r6c3",
            "r5c4",
            "r4c3",
            "r3c2",
        ),
    ),
    ("slow_thermo", ("r4c8", "r3c7", "r2c6", "r2c5", "r2c4", "r3c3")),
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


def get_slow_thermos_constraint_cores(num=SUDOKU_NUM, coreType=None):
    _check_num(num)
    return get_variant_constraint_cores(
        "slow_thermos",
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
        constraint_cores=get_slow_thermos_constraint_cores(num=num, coreType=coreType),
        startAssignment=startAssignment,
        initial_board=INITIAL_BOARD,
        include_initial_board=include_initial_board,
    )
