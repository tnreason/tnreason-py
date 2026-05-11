from __future__ import annotations

from typing import Tuple

from experiments.constraint_networks.sudoku_variant_forward_chaining import (
    Cell,
    SudokuVariantForwardChainer,
    SudokuVariantForwardChainingClosureEngine,
    all_cell_keys,
    black_dot_allowed,
    cell_key,
    edge_keys,
    parse_atom_key,
    parse_cell_key,
    red_line_allowed,
    sudoku_units,
    white_dot_allowed,
)


WHITE_DOTS: Tuple[Tuple[Cell, Cell], ...] = (
    ((0, 0), (1, 0)),
    ((0, 1), (0, 2)),
    ((0, 3), (1, 3)),
    ((0, 4), (1, 4)),
    ((1, 6), (2, 6)),
    ((3, 7), (3, 8)),
    ((4, 0), (4, 1)),
    ((4, 7), (4, 8)),
    ((5, 7), (5, 8)),
    ((6, 0), (6, 1)),
    ((6, 0), (7, 0)),
    ((6, 5), (7, 5)),
    ((6, 6), (6, 7)),
    ((7, 2), (8, 2)),
    ((7, 4), (7, 5)),
    ((8, 7), (8, 8)),
)


BLACK_DOTS: Tuple[Tuple[Cell, Cell], ...] = (
    ((1, 2), (2, 2)),
    ((2, 1), (3, 1)),
    ((3, 6), (3, 7)),
    ((7, 7), (8, 7)),
    ((8, 1), (8, 2)),
)


RED_LINE_EDGES: Tuple[Tuple[Cell, Cell], ...] = (
    ((2, 1), (2, 2)),
    ((2, 2), (2, 3)),
    ((2, 3), (2, 4)),
    ((2, 4), (2, 5)),
    ((2, 5), (3, 6)),
    ((3, 6), (4, 7)),
    ((4, 7), (5, 6)),
    ((6, 5), (6, 4)),
    ((6, 4), (6, 3)),
    ((6, 3), (6, 2)),
    ((6, 2), (6, 1)),
    ((2, 3), (3, 3)),
    ((3, 3), (4, 2)),
    ((4, 2), (5, 1)),
    ((5, 1), (5, 0)),
    ((3, 0), (3, 1)),
    ((3, 1), (4, 2)),
    ((4, 2), (5, 3)),
    ((5, 3), (6, 3)),
)


class SakanaFishForwardChainer(SudokuVariantForwardChainer):
    def __init__(self, num: int = 3):
        super().__init__(
            num=num,
            white_dot_edges=WHITE_DOTS,
            black_dot_edges=BLACK_DOTS,
            parity_edges=RED_LINE_EDGES,
        )


class SakanaFishForwardChainingClosureEngine(SudokuVariantForwardChainingClosureEngine):
    def __init__(self, num: int = 3, **kwargs):
        super().__init__(chainer=SakanaFishForwardChainer(num=num), num=num, **kwargs)
