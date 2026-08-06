from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


DIGITS = tuple(range(1, 10))
Cell = int
Domains = list[set[int]]


@dataclass
class TraceEvent:
    step: int
    kind: str
    message: str
    depth: int = 0
    board_id: str | None = None
    cell: str | None = None
    value: int | None = None
    before: list[int] | None = None
    after: list[int] | None = None
    removed: list[int] = field(default_factory=list)
    reason: str | None = None
    board: str | None = None


class SudokuReasoningTrace:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []
        self._next_board_id = 1

    def add(
        self,
        kind: str,
        message: str,
        *,
        depth: int = 0,
        board_id: str | None = None,
        cell: int | None = None,
        value: int | None = None,
        before: Iterable[int] | None = None,
        after: Iterable[int] | None = None,
        removed: Iterable[int] = (),
        reason: str | None = None,
        board: str | None = None,
    ) -> None:
        self.events.append(
            TraceEvent(
                step=len(self.events) + 1,
                kind=kind,
                message=message,
                depth=depth,
                board_id=board_id,
                cell=cell_name(cell) if cell is not None else None,
                value=value,
                before=sorted(before) if before is not None else None,
                after=sorted(after) if after is not None else None,
                removed=sorted(removed),
                reason=reason,
                board=board,
            )
        )

    def new_board_id(self) -> str:
        board_id = f"B{self._next_board_id}"
        self._next_board_id += 1
        return board_id

    def text(self) -> str:
        lines = []
        for event in self.events:
            prefix = f"{event.step:04d}"
            if event.board_id:
                prefix += f" {event.board_id}"
            indent = "  " * event.depth
            lines.append(f"{prefix} {indent}{event.message}")
        return "\n".join(lines) + ("\n" if lines else "")

    def write_text(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.text(), encoding="utf-8")

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")


@dataclass
class SudokuTraceResult:
    solved: bool
    board: str
    trace: SudokuReasoningTrace


def solve_with_trace(board: str) -> SudokuTraceResult:
    domains, givens = domains_from_board(board)
    trace = SudokuReasoningTrace()
    root_id = trace.new_board_id()
    for cell in sorted(givens):
        trace.add(
            "given",
            f"Given: {cell_name(cell)} = {next(iter(domains[cell]))}.",
            board_id=root_id,
            cell=cell,
            value=next(iter(domains[cell])),
            after=domains[cell],
            reason="initial clue",
            board=board_from_domains(domains),
        )

    solved_domains = _search(
        domains=domains,
        trace=trace,
        depth=0,
        board_id=root_id,
        reported_assignments=set(givens),
    )
    solved = solved_domains is not None and is_solved(solved_domains)
    return SudokuTraceResult(
        solved=solved,
        board=board_from_domains(solved_domains or domains),
        trace=trace,
    )


def _search(
    *,
    domains: Domains,
    trace: SudokuReasoningTrace,
    depth: int,
    board_id: str,
    reported_assignments: set[int],
) -> Domains | None:
    contradiction = propagate_to_fixpoint(
        domains=domains,
        trace=trace,
        depth=depth,
        board_id=board_id,
        reported_assignments=reported_assignments,
    )
    if contradiction:
        return None

    if is_solved(domains):
        trace.add(
            "solved",
            f"Solved board {board_id}.",
            depth=depth,
            board_id=board_id,
            board=board_from_domains(domains),
        )
        return domains

    branch_cell = min(
        (cell for cell in range(81) if len(domains[cell]) > 1),
        key=lambda cell: (len(domains[cell]), cell),
    )
    candidates = sorted(domains[branch_cell])
    trace.add(
        "branch_point",
        (
            f"Branch on {cell_name(branch_cell)} with candidates "
            f"{format_values(candidates)}."
        ),
        depth=depth,
        board_id=board_id,
        cell=branch_cell,
        before=domains[branch_cell],
        board=board_from_domains(domains),
    )

    for value in candidates:
        child_domains = copy_domains(domains)
        child_board_id = trace.new_board_id()
        child_domains[branch_cell] = {value}
        child_reported = set(reported_assignments)
        child_reported.add(branch_cell)
        trace.add(
            "guess",
            (
                f"Guess: assume {cell_name(branch_cell)} = {value} "
                f"from board {board_id}; continue as {child_board_id}."
            ),
            depth=depth,
            board_id=child_board_id,
            cell=branch_cell,
            value=value,
            before=domains[branch_cell],
            after={value},
            reason="minimum remaining candidates",
            board=board_from_domains(child_domains),
        )
        solved_domains = _search(
            domains=child_domains,
            trace=trace,
            depth=depth + 1,
            board_id=child_board_id,
            reported_assignments=child_reported,
        )
        if solved_domains is not None:
            return solved_domains

        trace.add(
            "backtrack",
            (
                f"Backtrack: {cell_name(branch_cell)} = {value} led to a "
                f"contradiction. Restore board {board_id}."
            ),
            depth=depth,
            board_id=board_id,
            cell=branch_cell,
            value=value,
            reason="failed branch",
            board=board_from_domains(domains),
        )

    trace.add(
        "contradiction",
        (
            f"Contradiction: every candidate for {cell_name(branch_cell)} "
            f"failed on board {board_id}."
        ),
        depth=depth,
        board_id=board_id,
        cell=branch_cell,
        before=domains[branch_cell],
        reason="all branches failed",
        board=board_from_domains(domains),
    )
    return None


def propagate_to_fixpoint(
    *,
    domains: Domains,
    trace: SudokuReasoningTrace,
    depth: int,
    board_id: str,
    reported_assignments: set[int],
) -> bool:
    changed = True
    while changed:
        changed = False
        for unit_name, unit in UNITS:
            for assigned_cell in unit:
                if len(domains[assigned_cell]) != 1:
                    continue
                digit = next(iter(domains[assigned_cell]))
                for peer in unit:
                    if peer == assigned_cell or digit not in domains[peer]:
                        continue
                    before = set(domains[peer])
                    domains[peer].remove(digit)
                    changed = True
                    trace.add(
                        "eliminate",
                        (
                            f"{cell_name(assigned_cell)} = {digit} implies "
                            f"{cell_name(peer)} != {digit} because both cells "
                            f"are in {unit_name}."
                        ),
                        depth=depth,
                        board_id=board_id,
                        cell=peer,
                        value=digit,
                        before=before,
                        after=domains[peer],
                        removed={digit},
                        reason=f"unit exclusion in {unit_name}",
                        board=board_from_domains(domains),
                    )
                    if not domains[peer]:
                        trace.add(
                            "contradiction",
                            (
                                f"Contradiction: {cell_name(peer)} has no "
                                f"remaining values after removing {digit}."
                            ),
                            depth=depth,
                            board_id=board_id,
                            cell=peer,
                            before=before,
                            after=domains[peer],
                            removed={digit},
                            reason=f"unit exclusion in {unit_name}",
                            board=board_from_domains(domains),
                        )
                        return True
                    _record_singleton(
                        domains,
                        trace,
                        depth,
                        board_id,
                        peer,
                        reported_assignments,
                    )

            for digit in DIGITS:
                places = [cell for cell in unit if digit in domains[cell]]
                if not places:
                    trace.add(
                        "contradiction",
                        (
                            f"Contradiction: {unit_name} has no possible "
                            f"position for digit {digit}."
                        ),
                        depth=depth,
                        board_id=board_id,
                        value=digit,
                        reason=f"hidden-single check in {unit_name}",
                        board=board_from_domains(domains),
                    )
                    return True
                if len(places) == 1 and len(domains[places[0]]) > 1:
                    cell = places[0]
                    before = set(domains[cell])
                    domains[cell] = {digit}
                    changed = True
                    reported_assignments.add(cell)
                    trace.add(
                        "hidden_single",
                        (
                            f"In {unit_name}, digit {digit} can only go in "
                            f"{cell_name(cell)}, so {cell_name(cell)} = {digit}."
                        ),
                        depth=depth,
                        board_id=board_id,
                        cell=cell,
                        value=digit,
                        before=before,
                        after=domains[cell],
                        removed=before - domains[cell],
                        reason=f"only possible position for {digit} in {unit_name}",
                        board=board_from_domains(domains),
                    )
    return False


def _record_singleton(
    domains: Domains,
    trace: SudokuReasoningTrace,
    depth: int,
    board_id: str,
    cell: Cell,
    reported_assignments: set[int],
) -> None:
    if len(domains[cell]) != 1 or cell in reported_assignments:
        return
    value = next(iter(domains[cell]))
    excluded = [digit for digit in DIGITS if digit != value]
    reported_assignments.add(cell)
    trace.add(
        "assign",
        (
            f"{cell_name(cell)} != {format_values(excluded)}, therefore "
            f"{cell_name(cell)} = {value}."
        ),
        depth=depth,
        board_id=board_id,
        cell=cell,
        value=value,
        after=domains[cell],
        removed=excluded,
        reason="only remaining value",
        board=board_from_domains(domains),
    )


def domains_from_board(board: str) -> tuple[Domains, set[int]]:
    cleaned = "".join(board.split())
    if len(cleaned) != 81:
        raise ValueError(f"Expected an 81-character Sudoku board, got {len(cleaned)}")
    domains: Domains = []
    givens: set[int] = set()
    for index, char in enumerate(cleaned):
        if char in ".0":
            domains.append(set(DIGITS))
            continue
        if char not in "123456789":
            raise ValueError(f"Invalid board character {char!r} at index {index}")
        domains.append({int(char)})
        givens.add(index)
    return domains, givens


def board_from_domains(domains: Domains) -> str:
    return "".join(
        str(next(iter(domain))) if len(domain) == 1 else "."
        for domain in domains
    )


def copy_domains(domains: Domains) -> Domains:
    return [set(domain) for domain in domains]


def is_solved(domains: Domains) -> bool:
    return all(len(domain) == 1 for domain in domains)


def cell_name(cell: Cell) -> str:
    row, col = divmod(cell, 9)
    return f"r{row + 1}c{col + 1}"


def format_values(values: Iterable[int]) -> str:
    return "{" + ",".join(str(value) for value in sorted(values)) + "}"


def _unit_name(kind: str, index: int) -> str:
    if kind == "row":
        return f"row {index + 1}"
    if kind == "column":
        return f"column {index + 1}"
    row = index // 3
    col = index % 3
    return f"box r{3 * row + 1}-r{3 * row + 3}, c{3 * col + 1}-c{3 * col + 3}"


def _build_units() -> tuple[tuple[str, tuple[Cell, ...]], ...]:
    units: list[tuple[str, tuple[Cell, ...]]] = []
    for row in range(9):
        units.append((_unit_name("row", row), tuple(row * 9 + col for col in range(9))))
    for col in range(9):
        units.append((_unit_name("column", col), tuple(row * 9 + col for row in range(9))))
    for box in range(9):
        box_row = box // 3
        box_col = box % 3
        cells = tuple(
            (box_row * 3 + row) * 9 + (box_col * 3 + col)
            for row in range(3)
            for col in range(3)
        )
        units.append((_unit_name("box", box), cells))
    return tuple(units)


UNITS = _build_units()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve a standard Sudoku and write an understandable reasoning trace."
    )
    parser.add_argument("board", help="81-character board using digits and . or 0 for blanks.")
    parser.add_argument(
        "--text-output",
        type=Path,
        help="Optional human-readable trace path.",
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        help="Optional structured JSONL trace path.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = solve_with_trace(args.board)
    print(result.board)
    print(f"solved={result.solved}, steps={len(result.trace.events)}")
    if args.text_output:
        result.trace.write_text(args.text_output)
        print(f"Wrote {args.text_output}")
    if args.jsonl_output:
        result.trace.write_jsonl(args.jsonl_output)
        print(f"Wrote {args.jsonl_output}")
    return 0 if result.solved else 1


if __name__ == "__main__":
    raise SystemExit(main())
