from demonstrations.sudoku.codex_agent.constraints.builder import build_variant_constraint_network

PUZZLE_INDEX = 64
ROW_INDEX = 24
PUZZLE_ID = 'sxsm_bellsita_4f473dd679d17b8b7f54ce5fad21a8b5'
PUZZLE_TITLE = 'Uraninite'
AUTHOR = 'bellsita'
SUDOKU_NUM = 3
INITIAL_BOARD = '.................................................................................'
ANTI_KNIGHT = False
KILLER_CAGES = [
    [
        [
            'r8c1',
            'r9c1',
            'r9c2'
        ],
        22,
        True
    ],
    [
        [
            'r5c2',
            'r6c2',
            'r7c2'
        ],
        22,
        True
    ],
    [
        [
            'r8c3',
            'r8c4',
            'r8c5'
        ],
        22,
        True
    ],
    [
        [
            'r1c8',
            'r1c9',
            'r2c9'
        ],
        8,
        True
    ],
    [
        [
            'r2c5',
            'r2c6',
            'r2c7'
        ],
        8,
        True
    ],
    [
        [
            'r3c8',
            'r4c8',
            'r5c8'
        ],
        8,
        True
    ],
    [
        [
            'r6c3',
            'r6c4',
            'r7c4'
        ],
        20,
        True
    ],
    [
        [
            'r3c6',
            'r4c6',
            'r4c7'
        ],
        20,
        True
    ],
    [
        [
            'r1c3',
            'r1c4'
        ],
        10,
        True
    ],
    [
        [
            'r9c6',
            'r9c7'
        ],
        10,
        True
    ]
]
NO_REPEAT_CAGES = [
    [
        'r2c1',
        'r2c2',
        'r3c1',
        'r4c1'
    ],
    [
        'r6c9',
        'r7c9',
        'r8c8',
        'r8c9'
    ]
]
ARROWS = []
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
    'r4c1',
    'r2c2',
    'r1c6',
    'r9c4',
    'r8c8',
    'r6c9'
]
EVEN_CELLS = [
    'r6c4',
    'r5c4',
    'r4c6',
    'r5c6'
]
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
