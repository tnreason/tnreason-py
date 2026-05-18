from demonstrations.sudoku.examples.challenge100_helpers import (
    get_assignment_as_constraint_network_from_cores,
    get_variant_constraint_cores,
)


PUZZLE_INDEX = 99
SUDOKU_NUM = 3
INITIAL_BOARD = (
    "3............................7.....2.......................8...........6........5"
)
CAGES = ()
NO_REPEAT_CAGES = ()
ARROWS = ()
EDGE_MARKERS = ()
LINES = (
    ("thermo", ("r6c4", "r5c4", "r4c5", "r3c6", "r3c5", "r3c4", "r3c3")),
    ("thermo", ("r6c4", "r6c5", "r5c6", "r4c7", "r5c7", "r6c7", "r7c7")),
    ("thermo", ("r1c9", "r2c8", "r3c7")),
    ("thermo", ("r8c1", "r7c1", "r7c2", "r6c3")),
    ("thermo", ("r9c2", "r9c3", "r8c3", "r7c4")),
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


def get_mc_hammer_constraint_cores(num=SUDOKU_NUM, coreType=None):
    _check_num(num)
    return get_variant_constraint_cores(
        "mc_hammer",
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
        constraint_cores=get_mc_hammer_constraint_cores(num=num, coreType=coreType),
        startAssignment=startAssignment,
        initial_board=INITIAL_BOARD,
        include_initial_board=include_initial_board,
    )
