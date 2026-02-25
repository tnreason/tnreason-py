# Sudoku Agents

Constraint implementations with validation tests.

## Workflow

1. **Coding agent** reads natural language rules from `challenge100.csv` (qwen-code)
2. **Implement** constraints as tensor networks, given already existing implementations
3. **Write validation tests** that verify constraint behavior matches the rule (i.e. sampling)
4. **Run local checks**: `python validation/test_all_constraints.py`

## Constraints

| Constraint | Rule | File |
|------------|------|------|
| Anti-Knight | Knight's move positions ≠ same value | `knight_move.py` |
| German Whispers | Adjacent digits differ by ≥ 5 | `german_whispers.py` |
| Renban | Line = consecutive non-repeating set | `renban.py` |
| Red Line | Exactly one position is odd | `../constraints/red_line.py` |
| Black Dot | 1:2 ratio | `../constraints/black_dot.py` |
| White Dot | Consecutive digits | `../constraints/white_dot.py` |

## Tests

```bash
# Run all 46 validation tests
python validation/test_all_constraints.py
```

All tests verify: valid assignments → 1, invalid assignments → 0

We need further checks and a more detailed review of the correctness by expert  
