# Sudoku Variant Constraints

Tensor network implementations of Sudoku variant constraints extracted from the `challenge100.csv` benchmark dataset.

---

## Workflow

1. **Rule Extraction**: Used a coding agent to extract all constraint rules from `challenge100.csv` → `constraint_rules.yaml`
2. **Implementation**: Agent generated tensor network implementations for each constraint
3. **Validation**: Agent created comprehensive test suite sampling valid/invalid assignments for each rule

---

## Quick Test

```bash
# Option 1: Run the test script
./test_constraints.sh

# Option 2: Run validation suite directly
python validation/test_all_constraints.py

# Option 3: Test individual constraints
for f in constraints/*.py; do python "$f"; done
```

**Expected output:**
```
============================================================
TEST SUMMARY
============================================================
Passed: 106/106
Failed: 0/106
============================================================
✓ All tests passed!
```

---

## Implemented Constraints (16)

| Category | Constraints |
|----------|-------------|
| **Adjacent Pair** | White Dot, Black Dot, Red Line, V/X Markers, German Whispers, Non-Consecutive |
| **Line** | Thermometer, Slow Thermometer, Renban, Entropic, Nabner, Palindrome, Anti-Knight |
| **Region** | Killer Cage, Arrow |

Full details in [`constraint_rules.yaml`](constraint_rules.yaml)

---

## Structure

```
agents/
├── constraints/          # 13 constraint implementations
│   ├── thermometer.py    # Strictly increasing line
│   ├── vx_marker.py      # V (sum=5), X (sum=10)
│   ├── killer_cage.py    # Sum to target, no repeats
│   ├── arrow.py          # Circle = sum of arrows
│   ├── entropic_line.py  # Low/Med/High diversity
│   ├── nabner.py         # Unique + non-consecutive
│   ├── palindrome.py     # Symmetric sequence
│   └── ...
├── validation/
│   └── test_all_constraints.py  # 106 tests
├── constraint_rules.yaml        # Complete rule catalog
└── README.md
```

---

## Usage Example

```python
from constraints import thermometer, killer_cage, arrow

# Thermometer: strictly increasing
thermo = thermometer.cenc_thermometer_constraint(["r0c0", "r0c1", "r0c2"])
assert thermo[{"r0c0": 0, "r0c1": 1, "r0c2": 2}] == 1  # 1,2,3 ✓

# Killer Cage: sum to 6, no repeats
cage = killer_cage.cenc_killer_cage_constraint(["r1c0", "r1c1", "r1c2"], target_sum=6)
assert cage[{"r1c0": 0, "r1c1": 1, "r1c2": 2}] == 1  # 1+2+3=6 ✓

# Arrow: circle = sum of arrow cells
arrow_const = arrow.cenc_arrow_constraint("circle", ["a1", "a2"])
assert arrow_const[{"circle": 2, "a1": 0, "a2": 1}] == 1  # 3 = 1+2 ✓
```

---

## Implementation Pattern

Each constraint provides:
- `constraint_name()` — Engine-based (slice iterator)
- `cenc_constraint_name()` — Representation-based (function encoding)
- `generate_constraint_constraints()` — Helper for cell coordinates

---

## Remaining TODO (4)

External clue constraints requiring row/column context:
- Sandwich Sum
- Skyscraper
- Numbered Room
- Look-and-Say Cage

---

## References

- Benchmark: `demonstrations/sudoku/sudoku_bench/sample_data/challenge100.csv`
- Rules: [`constraint_rules.yaml`](constraint_rules.yaml)
