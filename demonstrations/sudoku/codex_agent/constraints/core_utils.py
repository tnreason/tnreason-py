from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence

from tnreason import engine, representation
from tnreason.engine.workload_to_pandas import PandasCore


_COORD_RE = re.compile(r"^r(-?\d+)c(-?\d+)$")


def side(num: int = 3) -> int:
    return num * num


def coord_to_rc(coord: str) -> tuple[int, int]:
    match = _COORD_RE.match(str(coord))
    if match is None:
        raise ValueError(f"Cannot parse Sudoku coordinate {coord!r}")
    return int(match.group(1)) - 1, int(match.group(2)) - 1


def rc_to_coord(row: int, col: int) -> str:
    return f"r{row + 1}c{col + 1}"


def inside_grid(coord: str, num: int = 3) -> bool:
    row, col = coord_to_rc(coord)
    board_side = side(num)
    return 0 <= row < board_side and 0 <= col < board_side


def cell_color(coord: str, num: int = 3) -> str:
    row, col = coord_to_rc(coord)
    board_side = side(num)
    if not (0 <= row < board_side and 0 <= col < board_side):
        raise ValueError(f"Coordinate {coord!r} is outside a {board_side}x{board_side} grid")
    r1, r2 = divmod(row, num)
    c1, c2 = divmod(col, num)
    return f"pos_{r1}_{r2}_{c1}_{c2}"


def cell_colors(coords: Sequence[str], num: int = 3) -> tuple[str, ...]:
    return tuple(cell_color(coord, num=num) for coord in coords)


def unique_coords(coords: Sequence[str]) -> tuple[str, ...]:
    seen = set()
    unique = []
    for coord in coords:
        if coord in seen:
            continue
        seen.add(coord)
        unique.append(coord)
    return tuple(unique)


def drop_closing_repeat(coords: Sequence[str]) -> tuple[str, ...]:
    coords = tuple(coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        return coords[:-1]
    return coords


def atom_color(coord: str, digit: int, num: int = 3) -> str:
    row, col = coord_to_rc(coord)
    r1, r2 = divmod(row, num)
    c1, c2 = divmod(col, num)
    return f"a_{r1}_{r2}_{c1}_{c2}_{digit}"


def table_core(
    name: str,
    colors: Sequence[str],
    allowed_tuples: Iterable[Sequence[int]],
    shape: Sequence[int] | None = None,
    num: int = 3,
) -> PandasCore:
    colors = tuple(colors)
    if shape is None:
        shape = tuple(side(num) for _ in colors)
    rows = [
        {color: int(value) for color, value in zip(colors, values)}
        for values in allowed_tuples
    ]
    return PandasCore(values=rows, colors=list(colors), name=name, shape=list(shape))


def unary_core(
    name: str,
    coord: str,
    allowed_values: Iterable[int],
    num: int = 3,
) -> PandasCore:
    color = cell_color(coord, num=num)
    return table_core(
        name,
        (color,),
        ((value,) for value in allowed_values),
        shape=(side(num),),
        num=num,
    )


def binary_core(
    name: str,
    left: str,
    right: str,
    predicate,
    num: int = 3,
) -> PandasCore:
    board_side = side(num)
    return table_core(
        name,
        cell_colors((left, right), num=num),
        (
            (left_value, right_value)
            for left_value in range(board_side)
            for right_value in range(board_side)
            if predicate(left_value, right_value)
        ),
        num=num,
    )


def nary_core(
    name: str,
    coords: Sequence[str],
    allowed_tuples: Iterable[Sequence[int]],
    num: int = 3,
) -> PandasCore:
    return table_core(name, cell_colors(coords, num=num), allowed_tuples, num=num)


def evidence_cores(start_assignment: Mapping[str, int], core_dict: Mapping[str, object]):
    dim_dict = engine.get_dimDict(core_dict)
    return {
        evidence_key: representation.create_basis_core(
            name=evidence_key,
            shape=[dim_dict[evidence_key]],
            colors=[evidence_key],
            numberTuple=[start_assignment[evidence_key]],
        )
        for evidence_key in start_assignment
        if evidence_key in dim_dict
    }


def initial_board_to_evidence(board: str, num: int = 3) -> dict[str, int]:
    from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence

    return initial_board_into_evidence(board, colNum=num, rowNum=num)


def merge_core_dicts(*core_dicts: Mapping[str, object]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for core_dict in core_dicts:
        merged.update(core_dict)
    return merged


def standard_membership_core(atom: str, category: str, category_size: int, position: int):
    return engine.create_from_slice_iterator(
        colors=[atom, category],
        shape=[2, category_size],
        sliceIterator=[
            (1, {atom: 0}),
            (-1, {atom: 0, category: position}),
            (1, {atom: 1, category: position}),
        ],
        name=f"{category}_{atom}",
    )


def variant_unit_cores(
    units: Sequence[Sequence[str]],
    num: int = 3,
    prefix: str = "variant_unit",
) -> dict[str, object]:
    cores = {}
    board_side = side(num)
    for unit_index, cells in enumerate(units):
        cells = tuple(cells)
        for digit in range(board_side):
            category = f"variant_unit_{prefix}_{unit_index}_{digit}"
            for position, coord in enumerate(cells):
                atom = atom_color(coord, digit, num=num)
                key = f"{category}_{position}"
                cores[key] = standard_membership_core(atom, category, len(cells), position)
    return cores
