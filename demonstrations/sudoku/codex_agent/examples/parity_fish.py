from demonstrations.sudoku.codex_agent.constraints.builder import build_variant_constraint_network

PUZZLE_INDEX = 23
ROW_INDEX = 1
PUZZLE_ID = 'sxsm_MartySears_580c6fdbbba9bfb0e71ae19044f02d4c'
PUZZLE_TITLE = 'Parity Fish'
AUTHOR = 'Marty Sears'
SUDOKU_NUM = 3
INITIAL_BOARD = '.................................................................................'
ANTI_KNIGHT = False
KILLER_CAGES = []
NO_REPEAT_CAGES = []
ARROWS = []
EDGE_MARKERS = [
    [
        'white_dot',
        'r4c8',
        'r4c9'
    ],
    [
        'white_dot',
        'r5c8',
        'r5c9'
    ],
    [
        'white_dot',
        'r6c8',
        'r6c9'
    ],
    [
        'white_dot',
        'r5c1',
        'r5c2'
    ],
    [
        'white_dot',
        'r8c3',
        'r9c3'
    ],
    [
        'white_dot',
        'r7c1',
        'r8c1'
    ],
    [
        'white_dot',
        'r1c1',
        'r2c1'
    ],
    [
        'white_dot',
        'r7c7',
        'r7c8'
    ],
    [
        'white_dot',
        'r7c1',
        'r7c2'
    ],
    [
        'white_dot',
        'r9c8',
        'r9c9'
    ],
    [
        'white_dot',
        'r8c5',
        'r8c6'
    ],
    [
        'white_dot',
        'r1c4',
        'r2c4'
    ],
    [
        'white_dot',
        'r7c6',
        'r8c6'
    ],
    [
        'white_dot',
        'r2c7',
        'r3c7'
    ],
    [
        'white_dot',
        'r1c2',
        'r1c3'
    ],
    [
        'white_dot',
        'r1c5',
        'r2c5'
    ],
    [
        'black_dot',
        'r3c2',
        'r4c2'
    ],
    [
        'black_dot',
        'r4c7',
        'r4c8'
    ],
    [
        'black_dot',
        'r2c3',
        'r3c3'
    ],
    [
        'black_dot',
        'r9c2',
        'r9c3'
    ],
    [
        'black_dot',
        'r8c8',
        'r9c8'
    ]
]
STRICT_THERMOS = []
SLOW_THERMOS = []
RENBAN_LINES = []
WHISPER_LINES = []
DUTCH_WHISPER_LINES = []
ENTROPIC_LINES = []
MODULAR_LINES = []
EQUAL_SUM_LINES = []
ZIPPER_LINES = []
QUADRUPLES = []
PALINDROME_LINES = []
LITTLE_KILLERS = []
FIBONACCI_LINES = []
ARITHMETIC_SEQUENCE_LINES = []
LOCKOUT_LINES = []
REGION_SUM_LINES = []
INEQUALITIES = []
PARITY_EDGES = [
    [
        'r3c2',
        'r3c3'
    ],
    [
        'r3c3',
        'r3c4'
    ],
    [
        'r3c4',
        'r3c5'
    ],
    [
        'r3c5',
        'r3c6'
    ],
    [
        'r3c6',
        'r4c7'
    ],
    [
        'r4c7',
        'r5c8'
    ],
    [
        'r5c8',
        'r6c7'
    ],
    [
        'r6c7',
        'r7c6'
    ],
    [
        'r7c6',
        'r7c5'
    ],
    [
        'r7c5',
        'r7c4'
    ],
    [
        'r7c4',
        'r7c3'
    ],
    [
        'r7c3',
        'r7c2'
    ],
    [
        'r4c1',
        'r4c2'
    ],
    [
        'r4c2',
        'r5c3'
    ],
    [
        'r5c3',
        'r6c4'
    ],
    [
        'r6c4',
        'r7c4'
    ],
    [
        'r6c1',
        'r6c2'
    ],
    [
        'r6c2',
        'r5c3'
    ],
    [
        'r5c3',
        'r4c4'
    ],
    [
        'r4c4',
        'r3c4'
    ]
]
ODD_CELLS = []
EVEN_CELLS = []
CAGES = KILLER_CAGES
LINES = {
    "strict_thermos": STRICT_THERMOS,
    "slow_thermos": SLOW_THERMOS,
    "renban": RENBAN_LINES,
    "whisper": WHISPER_LINES,
    "dutch_whisper": DUTCH_WHISPER_LINES,
    "entropy": ENTROPIC_LINES,
    "modular": MODULAR_LINES,
    "equal_sum": EQUAL_SUM_LINES,
    "zipper": ZIPPER_LINES,
    "palindrome": PALINDROME_LINES,
    "fibonacci": FIBONACCI_LINES,
    "arithmetic_sequence": ARITHMETIC_SEQUENCE_LINES,
    "lockout": LOCKOUT_LINES,
    "region_sum": REGION_SUM_LINES,
}
PARITY_CELLS = {"odd": ODD_CELLS, "even": EVEN_CELLS}


def get_assignment_as_constraint_network(num=SUDOKU_NUM, startAssignment=None, include_initial_board=True):
    if num != SUDOKU_NUM:
        raise ValueError(f"This example is for num={SUDOKU_NUM}, got {num}")
    return build_variant_constraint_network(
        num=num,
        start_assignment=startAssignment or {},
        include_initial_board=include_initial_board,
        initial_board=INITIAL_BOARD,
        anti_knight=ANTI_KNIGHT,
        killer_cages=KILLER_CAGES,
        no_repeat_cages=NO_REPEAT_CAGES,
        arrows=ARROWS,
        edge_markers=EDGE_MARKERS,
        strict_thermos=STRICT_THERMOS,
        slow_thermos=SLOW_THERMOS,
        renban_lines=RENBAN_LINES,
        whisper_lines=WHISPER_LINES,
        dutch_whisper_lines=DUTCH_WHISPER_LINES,
        entropic_lines=ENTROPIC_LINES,
        modular_lines=MODULAR_LINES,
        equal_sum_lines=EQUAL_SUM_LINES,
        zipper_lines=ZIPPER_LINES,
        quadruples=QUADRUPLES,
        palindrome_lines=PALINDROME_LINES,
        little_killers=LITTLE_KILLERS,
        fibonacci_lines=FIBONACCI_LINES,
        arithmetic_sequence_lines=ARITHMETIC_SEQUENCE_LINES,
        lockout_lines=LOCKOUT_LINES,
        region_sum_lines=REGION_SUM_LINES,
        inequalities=INEQUALITIES,
        parity_edges=PARITY_EDGES,
        odd_cells=ODD_CELLS,
        even_cells=EVEN_CELLS,
    )
