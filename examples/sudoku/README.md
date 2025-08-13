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