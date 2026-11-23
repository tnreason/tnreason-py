from demonstrations.sudoku.codex_agent.constraints.builder import build_variant_constraint_network

PUZZLE_INDEX = 41
ROW_INDEX = 12
PUZZLE_ID = 'sxsm_Neokart_d5260b08ac3932463f0aa0e660e15106'
PUZZLE_TITLE = '\xc1rbol'
AUTHOR = 'Neokart'
SUDOKU_NUM = 3
INITIAL_BOARD = '.................................................................................'
ANTI_KNIGHT = False
KILLER_CAGES = []
NO_REPEAT_CAGES = []
ARROWS = []
EDGE_MARKERS = [
    [
        'white_dot',
        'r6c5',
        'r6c6'
    ],
    [
        'white_dot',
        'r6c4',
        'r6c5'
    ],
    [
        'white_dot',
        'r3c5',
        'r3c6'
    ],
    [
        'white_dot',
        'r3c4',
        'r3c5'
    ],
    [
        'white_dot',
        'r8c3',
        'r8c4'
    ],
    [
        'black_dot',
        'r1c2',
        'r2c2'
    ],
    [
        'black_dot',
        'r1c8',
        'r2c8'
    ],
    [
        'black_dot',
        'r6c3',
        'r7c3'
    ],
    [
        'black_dot',
        'r6c7',
        'r7c7'
    ],
    [
        'black_dot',
        'r8c6',
        'r8c7'
    ]
]
STRICT_THERMOS = []
SLOW_THERMOS = [
    [
        'r7c5',
        'r6c6',
        'r6c7',
        'r5c8'
    ],
    [
        'r7c5',
        'r6c6',
        'r6c7',
        'r6c8'
    ],
    [
        'r7c5',
        'r6c4',
        'r6c3',
        'r5c2'
    ],
    [
        'r7c5',
        'r6c4',
        'r6c3',
        'r6c2'
    ],
    [
        'r7c5',
        'r6c5',
        'r5c5',
        'r4c6',
        'r4c7',
        'r3c8'
    ],
    [
        'r7c5',
        'r6c5',
        'r5c5',
        'r4c4',
        'r4c3',
        'r3c2'
    ],
    [
        'r7c5',
        'r6c5',
        'r5c5',
        'r4c5',
        'r3c6',
        'r2c7',
        'r1c8'
    ],
    [
        'r7c5',
        'r6c5',
        'r5c5',
        'r4c5',
        'r3c4',
        'r2c3',
        'r1c2'
    ],
    [
        'r7c5',
        'r6c5',
        'r5c5',
        'r4c5',
        'r3c5',
        'r2c5',
        'r1c5'
    ],
    [
        'r7c5',
        'r8c5',
        'r9c5'
    ],
    [
        'r7c5',
        'r8c5',
        'r9c4'
    ],
    [
        'r7c5',
        'r8c5',
        'r9c6'
    ],
    [
        'r7c5',
        'r8c5',
        'r8c6',
        'r9c7'
    ],
    [
        'r7c5',
        'r8c5',
        'r8c4',
        'r9c3'
    ]
]
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
PARITY_EDGES = []
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
