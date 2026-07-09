import re

from tnreason import representation
from tnreason.engine import get_dimDict

from demonstrations.sudoku.constraints.all_different import (
    all_different_pair_constraints,
)
from demonstrations.sudoku.constraints.anti_knight import all_anti_knight_constraints
from demonstrations.sudoku.constraints.arrow_sum import arrow_sum_constraint
from demonstrations.sudoku.constraints.balanced_line import (
    balanced_four_cell_line_constraint,
)
from demonstrations.sudoku.constraints.black_dot import black_dot_constraint
from demonstrations.sudoku.constraints.entropic_line import entropic_line_constraints
from demonstrations.sudoku.constraints.fibonacci_line import fibonacci_line_constraint
from demonstrations.sudoku.constraints.gray_dot import gray_dot_constraint
from demonstrations.sudoku.constraints.inequality import inequality_constraint
from demonstrations.sudoku.constraints.killer_cage import killer_cage_constraint
from demonstrations.sudoku.constraints.little_killer import little_killer_constraint
from demonstrations.sudoku.constraints.lockout_line import lockout_line_constraint
from demonstrations.sudoku.constraints.modular_line import modular_line_constraints
from demonstrations.sudoku.constraints.palindrome_line import (
    palindrome_pair_constraints,
)
from demonstrations.sudoku.constraints.parity_cell import (
    even_cell_constraint,
    odd_cell_constraint,
)
from demonstrations.sudoku.constraints.quadruple import quadruple_constraint
from demonstrations.sudoku.constraints.red_dot import red_dot_constraint
from demonstrations.sudoku.constraints.red_line import red_line_constraint
from demonstrations.sudoku.constraints.region_sum_line import (
    region_sum_line_constraints,
)
from demonstrations.sudoku.constraints.renban_line import renban_line_constraint
from demonstrations.sudoku.constraints.sequence_line import sequence_line_constraint
from demonstrations.sudoku.constraints.thermometer_line import (
    slow_thermometer_line_constraints,
    thermometer_line_constraints,
)
from demonstrations.sudoku.constraints.whisper_line import whisper_line_constraints
from demonstrations.sudoku.constraints.white_dot import white_dot_constraint
from demonstrations.sudoku.constraints.xv import v_constraint, x_constraint
from demonstrations.sudoku.constraints.zipper_line import zipper_line_constraints
from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence
from experiments.constraint_networks.sudoku_tests.standard_constraints import (
    get_sudoku_constraint_network,
)


_COORD_RE = re.compile(r"^r(-?\d+)c(-?\d+)$")


def parse_coord(coord):
    match = _COORD_RE.match(str(coord))
    if match is None:
        raise ValueError(f"Cannot parse Sudoku coordinate {coord!r}")
    return int(match.group(1)) - 1, int(match.group(2)) - 1


def cell_var(coord, num=3):
    row, col = parse_coord(coord)
    row_block, row_inner = divmod(row, num)
    col_block, col_inner = divmod(col, num)
    return f"pos_{row_block}_{row_inner}_{col_block}_{col_inner}"


def coords_to_vars(coords, num=3):
    return [cell_var(coord, num=num) for coord in coords]


def _drop_consecutive_duplicates(values):
    deduped = []
    for value in values:
        if not deduped or deduped[-1] != value:
            deduped.append(value)
    return deduped


def _unique_line_vars(coords, num):
    values = _drop_consecutive_duplicates(coords_to_vars(coords, num=num))
    if len(values) > 1 and values[0] == values[-1]:
        values = values[:-1]
    unique = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        unique.append(value)
        seen.add(value)
    return unique


def _box_id(coord, num):
    row, col = parse_coord(coord)
    return row // num, col // num


def _split_line_by_box(coords, num):
    coords = _drop_consecutive_duplicates(list(coords))
    is_closed_loop = len(coords) > 1 and coords[0] == coords[-1]
    if is_closed_loop:
        coords = coords[:-1]

    segments = []
    segment_boxes = []
    current_box = None
    for coord in coords:
        next_box = _box_id(coord, num)
        if next_box != current_box:
            segments.append([])
            segment_boxes.append(next_box)
            current_box = next_box
        segments[-1].append(cell_var(coord, num=num))

    if is_closed_loop and len(segments) > 1 and segment_boxes[0] == segment_boxes[-1]:
        segments[0] = segments[-1] + segments[0]
        segments.pop()

    return [segment for segment in segments if segment]


def _put_constraint(constraints, key, core):
    candidate = key
    suffix = 2
    while candidate in constraints:
        candidate = f"{key}_{suffix}"
        suffix += 1
    constraints[candidate] = core


def _merge_constraints(constraints, new_constraints):
    for key, core in new_constraints.items():
        if key in constraints and key.startswith("oddInd_") and key.endswith("_core"):
            continue
        _put_constraint(constraints, key, core)


def _add_line(constraints, prefix, index, rule, coords, num, coreType):
    vars_ = _drop_consecutive_duplicates(coords_to_vars(coords, num=num))
    nary_vars = _unique_line_vars(coords, num=num)
    key_prefix = f"{prefix}_line_{index}_{rule}"

    if rule == "thermo":
        _merge_constraints(
            constraints,
            thermometer_line_constraints(
                vars_, sudokuNum=num, prefix=key_prefix, coreType=coreType
            ),
        )
    elif rule == "slow_thermo":
        _merge_constraints(
            constraints,
            slow_thermometer_line_constraints(
                vars_, sudokuNum=num, prefix=key_prefix, coreType=coreType
            ),
        )
    elif rule == "renban" and len(nary_vars) >= 2:
        _put_constraint(
            constraints,
            key_prefix,
            renban_line_constraint(nary_vars, sudokuNum=num, coreType=coreType),
        )
    elif rule == "whisper":
        _merge_constraints(
            constraints,
            whisper_line_constraints(
                vars_, sudokuNum=num, prefix=key_prefix, coreType=coreType
            ),
        )
    elif rule == "dutch_whisper":
        _merge_constraints(
            constraints,
            whisper_line_constraints(
                vars_,
                min_difference=4,
                sudokuNum=num,
                prefix=key_prefix,
                coreType=coreType,
            ),
        )
    elif rule == "balanced" and len(nary_vars) == 4:
        _put_constraint(
            constraints,
            key_prefix,
            balanced_four_cell_line_constraint(
                nary_vars, sudokuNum=num, coreType=coreType
            ),
        )
    elif rule == "entropic":
        _merge_constraints(
            constraints,
            entropic_line_constraints(
                vars_, sudokuNum=num, prefix=key_prefix, coreType=coreType
            ),
        )
    elif rule == "modular":
        _merge_constraints(
            constraints,
            modular_line_constraints(
                vars_, sudokuNum=num, prefix=key_prefix, coreType=coreType
            ),
        )
    elif rule == "palindrome":
        _merge_constraints(
            constraints,
            palindrome_pair_constraints(
                nary_vars, sudokuNum=num, prefix=key_prefix, coreType=coreType
            ),
        )
    elif rule == "region_sum":
        _merge_constraints(
            constraints,
            region_sum_line_constraints(
                _split_line_by_box(coords, num=num),
                sum_var=f"{key_prefix}_sum",
                sudokuNum=num,
                prefix=key_prefix,
                coreType=coreType,
            ),
        )
    elif rule == "zipper":
        _merge_constraints(
            constraints,
            zipper_line_constraints(
                nary_vars, sudokuNum=num, prefix=key_prefix, coreType=coreType
            ),
        )
    elif rule == "fibonacci" and len(nary_vars) >= 2:
        _put_constraint(
            constraints,
            key_prefix,
            fibonacci_line_constraint(nary_vars, sudokuNum=num, coreType=coreType),
        )
    elif rule == "sequence" and len(nary_vars) >= 2:
        _put_constraint(
            constraints,
            key_prefix,
            sequence_line_constraint(nary_vars, sudokuNum=num, coreType=coreType),
        )
    elif rule == "lockout" and len(nary_vars) >= 3:
        _put_constraint(
            constraints,
            key_prefix,
            lockout_line_constraint(nary_vars, sudokuNum=num, coreType=coreType),
        )


def get_variant_constraint_cores(
    prefix,
    num=3,
    cages=(),
    no_repeat_cages=(),
    arrows=(),
    edge_markers=(),
    lines=(),
    quadruples=(),
    parity_cells=(),
    little_killers=(),
    inequalities=(),
    anti_knight=False,
    coreType=None,
):
    constraints = {}

    for index, (coords, target_sum) in enumerate(cages):
        _put_constraint(
            constraints,
            f"{prefix}_cage_{index}",
            killer_cage_constraint(
                coords_to_vars(coords, num=num),
                target_sum,
                sudokuNum=num,
                coreType=coreType,
            ),
        )

    for index, coords in enumerate(no_repeat_cages):
        _merge_constraints(
            constraints,
            all_different_pair_constraints(
                coords_to_vars(coords, num=num),
                sudokuNum=num,
                prefix=f"{prefix}_no_repeat_cage_{index}",
                coreType=coreType,
            ),
        )

    for index, (circle, arrow_coords) in enumerate(arrows):
        _put_constraint(
            constraints,
            f"{prefix}_arrow_{index}",
            arrow_sum_constraint(
                cell_var(circle, num=num),
                coords_to_vars(arrow_coords, num=num),
                sudokuNum=num,
                coreType=coreType,
            ),
        )

    for index, (rule, coord1, coord2) in enumerate(edge_markers):
        posVar1 = cell_var(coord1, num=num)
        posVar2 = cell_var(coord2, num=num)
        key = f"{prefix}_{rule}_{index}"
        if rule == "white_dot":
            _put_constraint(
                constraints, key, white_dot_constraint(posVar1, posVar2, sudokuNum=num)
            )
        elif rule == "black_dot":
            _put_constraint(
                constraints, key, black_dot_constraint(posVar1, posVar2, sudokuNum=num)
            )
        elif rule == "red_dot":
            _put_constraint(
                constraints,
                key,
                red_dot_constraint(posVar1, posVar2, sudokuNum=num, coreType=coreType),
            )
        elif rule == "gray_dot":
            _put_constraint(
                constraints,
                key,
                gray_dot_constraint(posVar1, posVar2, sudokuNum=num, coreType=coreType),
            )
        elif rule == "v":
            _put_constraint(
                constraints,
                key,
                v_constraint(posVar1, posVar2, sudokuNum=num, coreType=coreType),
            )
        elif rule == "x":
            _put_constraint(
                constraints,
                key,
                x_constraint(posVar1, posVar2, sudokuNum=num, coreType=coreType),
            )
        elif rule == "red_line":
            _merge_constraints(
                constraints, red_line_constraint(posVar1, posVar2, sudokuNum=num)
            )

    for index, (rule, coords) in enumerate(lines):
        _add_line(constraints, prefix, index, rule, coords, num, coreType)

    for index, (coords, required_digits) in enumerate(quadruples):
        _put_constraint(
            constraints,
            f"{prefix}_quadruple_{index}",
            quadruple_constraint(
                coords_to_vars(coords, num=num),
                required_digits,
                sudokuNum=num,
                coreType=coreType,
            ),
        )

    for index, (parity, coord) in enumerate(parity_cells):
        factory = odd_cell_constraint if parity == "odd" else even_cell_constraint
        _put_constraint(
            constraints,
            f"{prefix}_{parity}_{index}",
            factory(cell_var(coord, num=num), sudokuNum=num, coreType=coreType),
        )

    for index, (coords, target_sum) in enumerate(little_killers):
        _put_constraint(
            constraints,
            f"{prefix}_little_killer_{index}",
            little_killer_constraint(
                coords_to_vars(coords, num=num),
                target_sum,
                sudokuNum=num,
                coreType=coreType,
            ),
        )

    for index, (coord1, relation, coord2) in enumerate(inequalities):
        _put_constraint(
            constraints,
            f"{prefix}_inequality_{index}",
            inequality_constraint(
                cell_var(coord1, num=num),
                cell_var(coord2, num=num),
                relation=relation,
                sudokuNum=num,
                coreType=coreType,
            ),
        )

    if anti_knight:
        _merge_constraints(
            constraints,
            all_anti_knight_constraints(
                row_count=num**2,
                col_count=num**2,
                cell_var=lambda row, col: cell_var(f"r{row + 1}c{col + 1}", num=num),
                sudokuNum=num,
                coreType=coreType,
            ),
        )

    return constraints


def get_assignment_as_constraint_network_from_cores(
    num,
    constraint_cores,
    startAssignment=None,
    initial_board=None,
    include_initial_board=True,
):
    constraint_cores = {
        **get_sudoku_constraint_network(num=num),
        **constraint_cores,
    }

    assignment = {}
    if include_initial_board and initial_board is not None:
        assignment.update(
            initial_board_into_evidence(initial_board, colNum=num, rowNum=num)
        )
    if startAssignment:
        assignment.update(startAssignment)

    dim_dict = get_dimDict(constraint_cores)
    evidence_cores = {
        evidenceKey: representation.create_basis_core(
            name=evidenceKey,
            shape=[dim_dict[evidenceKey]],
            colors=[evidenceKey],
            numberTuple=[assignment[evidenceKey]],
        )
        for evidenceKey in assignment
    }
    return {
        **constraint_cores,
        **evidence_cores,
    }
