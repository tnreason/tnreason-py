# tnreason for Sudoku

We here demonstrate the usage of `tnreason` for solving Sudoku puzzles using a tensor network approach.

## Background

Sudoku is an instance of a Constraint Satisfaction Problem, which are in more generality representable by boolean tensor networks.

## Files 

- `representation.py` spezifies the Sudoku rules as categorical constraints.
- `visualization.py` visualized the Sudoku puzzle as arrays.
- `solution.py` prepares a solution based on an Expectation-Propagation algorithm.

## Examples

Examples can be found in `examples/sudoku/examples/`:
- `n=2.py` a simple example of a (2x2)x(2x2) Sudoku puzzle, which is solved by single atom-categorical message passing.
- `n=3.py` a more complex example of a (3x3)x(3x3) Sudoku puzzle

We further apply the algorithms on the [Sudoku-Bench Dataset](https://huggingface.co/datasets/SakanaAI/Sudoku-Bench) of Sakana AI in `examples/sudoku/sudoku_bench`.

## Optional Sudoku clusters for MAC

Generic MAC and the benchmark MAC adapters do not automatically select Sudoku
clusters. The helpers below are separate, explicit experiment utilities. Use
`network.solve_mac(domains=domains)` for generic MAC without these helpers.

For a `ConstraintNetwork` using the standard categorical Sudoku encoding:

```python
from demonstrations.sudoku.inferenceClusters.mac import prepare_sudoku_state

state = prepare_sudoku_state(network, domains=domains, num=3, pairs=True, locked=True)
result = network.solve_mac(state=state)
```

Use `num=2` for 4x4 Sudoku; omit `domains` when givens are already cores.
With `pairs=True`, preprocessing selects naked/hidden pairs after root propagation
and contracts their eight original links into four-variable Boolean tensors.
With `locked=True`, affected row–box and column–box intersections are checked
after each GAC closure, including during search. A two-variable Boolean summary
is added only when pointing or claiming can remove candidates. Blank Sudoku
adds none. Contracted tensors are cached for reuse, keyed by their actual
Boolean supports and local tensor topology. Pairs default on in this
helper; locked candidates default off. Either can be enabled independently.
The propagated domains and contraction caches are reused for search; only the
new cores and affected constraints are queued. Summaries do not include the
selection domains. Branch additions are removed on backtracking, while their
cached tensors can be reused in later branches. Original cores remain unchanged;
generic MAC has no Sudoku rules.
Pairs first appearing later in search are not selected automatically.
Already solved or contradictory roots do not need extra summaries.

`inferenceClusters/CODEGEN_PROMPT.txt` contains a code-generation task for
extending the standalone cluster library while keeping MAC and its benchmark
adapters independent of Sudoku heuristics.

## Standalone rule library

The expanded [inference cluster library](inferenceClusters/README.md) describes
finite rule patterns and checks their consequences by exact Boolean contraction
of the original tensors. Select rule families explicitly through its registry;
importing the library does not enable Sudoku heuristics in generic MAC.

The comparison experiment accepts a Nikoli100 parquet dataset and writes a new
local result directory:

```bash
python -m experiments.sudoku_rule_clusters \
  --dataset /path/to/nikoli100.parquet \
  --families locked_candidates basic_fish --limit 10
```

Reading parquet requires a pandas parquet engine such as PyArrow. See
`python -m experiments.sudoku_rule_clusters --help` for scan, contraction, and
search budgets.
