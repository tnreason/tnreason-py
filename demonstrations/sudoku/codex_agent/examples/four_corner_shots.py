from demonstrations.sudoku.codex_agent.constraints.builder import build_variant_constraint_network

PUZZLE_INDEX = 34
ROW_INDEX = 10
PUZZLE_ID = 'fpuzzled2b81e418308a21db8bb56aaab4b778a'
PUZZLE_TITLE = 'Four Corner Shots'
AUTHOR = 'noswald'
SUDOKU_NUM = 3
INITIAL_BOARD = '.................................................................................'
ANTI_KNIGHT = False
KILLER_CAGES = [
    [
        [
            'r1c1',
            'r1c2',
            'r2c1',
            'r2c2'
        ],
        17,
        True
    ],
    [
        [
            'r1c5',
            'r1c6',
            'r2c5',
            'r2c6'
        ],
        11,
        True
    ],
    [
        [
            'r1c8',
            'r1c9',
            'r2c8',
            'r2c9'
        ],
        17,
        True
    ],
    [
        [
            'r8c1',
            'r8c2',
            'r9c1',
            'r9c2'
        ],
        20,
        True
    ],
    [
        [
            'r8c5',
            'r8c6',
            'r9c5',
            'r9c6'
        ],
        10,
        True
    ],
    [
        [
            'r8c8',
            'r8c9',
            'r9c8',
            'r9c9'
        ],
        15,
        True
    ]
]
NO_REPEAT_CAGES = []
ARROWS = [
    [
        'r1c1',
        'r2c2',
        'r3c2',
        'r4c2'
    ],
    [
        'r9c2',
        'r8c1',
        'r7c1',
        'r6c1'
    ],
    [
        'r2c8',
        'r2c9',
        'r3c9',
        'r4c9'
    ],
    [
        'r8c9',
        'r8c8',
        'r7c8',
        'r6c8'
    ],
    [
        'r5c2',
        'r4c3',
        'r5c3',
        'r6c3'
    ],
    [
        'r5c5',
        'r4c4',
        'r5c4',
        'r6c4'
    ],
    [
        'r5c8',
        'r4c7',
        'r5c7',
        'r6c7'
    ]
]
EDGE_MARKERS = []
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
PARITY_EDGES = []
ODD_CELLS = [
    'r4c4'
]
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
