# AlphaReason

`experiments/alphareason` is a generic Alpha-style search framework on top of TNReason.

It searches over assignments to a designated set of categorical target features while using TNReason
message passing as the closure step after each action.

Core ideas:

- tasks specify a network, initial evidence, target features, and optional solution assignments
- actions assign one value to one undecided categorical feature
- closure is still done by TNReason message passing
- MCTS operates on closed states
- policy targets can be extracted from solution assignments by taking all solution-consistent next actions

Sudoku is included as the first adapter under `experiments/alphareason/tasks/sudoku.py`.


