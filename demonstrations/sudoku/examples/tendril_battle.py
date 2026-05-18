from demonstrations.sudoku.examples.challenge100_helpers import (
    get_assignment_as_constraint_network_from_cores,
    get_variant_constraint_cores,
)


PUZZLE_INDEX = 29
SUDOKU_NUM = 3
INITIAL_BOARD = (
    "................................................................................."
)
CAGES = ()
NO_REPEAT_CAGES = ()
ARROWS = ()
EDGE_MARKERS = ()
LINES = (
    ("thermo", ("r7c1", "r6c1", "r5c1", "r4c1", "r4c2", "r3c3")),
    ("thermo", ("r7c1", "r6c2", "r5c3", "r5c4", "r5c5")),
    ("thermo", ("r7c1", "r7c2", "r7c3", "r7c4", "r7c5")),
    ("thermo", ("r7c1", "r8c1", "r8c2", "r9c3", "r9c4")),
    ("thermo", ("r2c7", "r3c8", "r4c8", "r5c9", "r6c9", "r6c8", "r6c7")),
    ("thermo", ("r2c7", "r3c6", "r3c5")),
    ("thermo", ("r2c7", "r2c6", "r1c5", "r2c4", "r2c3", "r2c2")),
    ("thermo", ("r2c7", "r3c7", "r4c7", "r5c6", "r6c6")),
)
QUADRUPLES = ()
PARITY_CELLS = ()
LITTLE_KILLERS = (
    (("r7c1", "r8c2", "r9c3"), 9),
    (("r5c9", "r4c8", "r3c7", "r2c6", "r1c5"), 15),
)
INEQUALITIES = ()
ANTI_KNIGHT = False


def _check_num(num):
    if num != SUDOKU_NUM:
        raise ValueError(
            f"Puzzle {PUZZLE_INDEX} is represented with num={SUDOKU_NUM}, not num={num}."
        )


def get_tendril_battle_constraint_cores(num=SUDOKU_NUM, coreType=None):
    _check_num(num)
    return get_variant_constraint_cores(
        "tendril_battle",
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
        constraint_cores=get_tendril_battle_constraint_cores(
            num=num, coreType=coreType
        ),
        startAssignment=startAssignment,
        initial_board=INITIAL_BOARD,
        include_initial_board=include_initial_board,
    )
