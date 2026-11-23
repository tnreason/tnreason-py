from demonstrations.sudoku.reasoning_trace import solve_with_trace


def test_reasoning_trace_solves_easy_standard_sudoku():
    board = (
        "53..7...."
        "6..195..."
        ".98....6."
        "8...6...3"
        "4..8.3..1"
        "7...2...6"
        ".6....28."
        "...419..5"
        "....8..79"
    )

    result = solve_with_trace(board)

    assert result.solved
    assert result.board == (
        "534678912"
        "672195348"
        "198342567"
        "859761423"
        "426853791"
        "713924856"
        "961537284"
        "287419635"
        "345286179"
    )
    assert any(event.kind == "eliminate" for event in result.trace.events)
    assert any(event.kind in {"assign", "hidden_single"} for event in result.trace.events)


def test_reasoning_trace_records_guess_and_backtrack():
    # This board is intentionally underconstrained enough that the solver must
    # branch; the trace should expose the search decisions.
    board = "." * 81

    result = solve_with_trace(board)

    assert result.solved
    assert any(event.kind == "guess" for event in result.trace.events)
