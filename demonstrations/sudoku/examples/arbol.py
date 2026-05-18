from demonstrations.sudoku.examples.challenge100_helpers import (
    get_assignment_as_constraint_network_from_cores,
    get_variant_constraint_cores,
)


PUZZLE_INDEX = 41
SUDOKU_NUM = 3
INITIAL_BOARD = (
    "................................................................................."
)
CAGES = ()
NO_REPEAT_CAGES = ()
ARROWS = ()
EDGE_MARKERS = (
    ("white_dot", "r6c5", "r6c6"),
    ("white_dot", "r6c4", "r6c5"),
    ("white_dot", "r3c5", "r3c6"),
    ("white_dot", "r3c4", "r3c5"),
    ("white_dot", "r8c3", "r8c4"),
    ("black_dot", "r1c2", "r2c2"),
    ("black_dot", "r1c8", "r2c8"),
    ("black_dot", "r6c3", "r7c3"),
    ("black_dot", "r6c7", "r7c7"),
    ("black_dot", "r8c6", "r8c7"),
)
LINES = (
    ("slow_thermo", ("r7c5", "r6c6", "r6c7", "r5c8")),
    ("slow_thermo", ("r7c5", "r6c6", "r6c7", "r6c8")),
    ("slow_thermo", ("r7c5", "r6c4", "r6c3", "r5c2")),
    ("slow_thermo", ("r7c5", "r6c4", "r6c3", "r6c2")),
    ("slow_thermo", ("r7c5", "r6c5", "r5c5", "r4c6", "r4c7", "r3c8")),
    ("slow_thermo", ("r7c5", "r6c5", "r5c5", "r4c4", "r4c3", "r3c2")),
    ("slow_thermo", ("r7c5", "r6c5", "r5c5", "r4c5", "r3c6", "r2c7", "r1c8")),
    ("slow_thermo", ("r7c5", "r6c5", "r5c5", "r4c5", "r3c4", "r2c3", "r1c2")),
    ("slow_thermo", ("r7c5", "r6c5", "r5c5", "r4c5", "r3c5", "r2c5", "r1c5")),
    ("slow_thermo", ("r7c5", "r8c5", "r9c5")),
    ("slow_thermo", ("r7c5", "r8c5", "r9c4")),
    ("slow_thermo", ("r7c5", "r8c5", "r9c6")),
    ("slow_thermo", ("r7c5", "r8c5", "r8c6", "r9c7")),
    ("slow_thermo", ("r7c5", "r8c5", "r8c4", "r9c3")),
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


def get_arbol_constraint_cores(num=SUDOKU_NUM, coreType=None):
    _check_num(num)
    return get_variant_constraint_cores(
        "arbol",
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
        constraint_cores=get_arbol_constraint_cores(num=num, coreType=coreType),
        startAssignment=startAssignment,
        initial_board=INITIAL_BOARD,
        include_initial_board=include_initial_board,
    )
