# AlphaSudoku

`experiments/alphasudoku` is a new Alpha-style search architecture for Sudoku built on top of `tnreason`.

It is intended to replace the old `backtrack_caNetwork.py` style of mixing:

- propagation
- inference-cluster extension
- guessing
- search control

in one file.

Instead, this package separates:

- deterministic TNReason-based state closure
- legal action generation
- MCTS search
- policy/value heuristics


## Goal

The goal is to search over Sudoku states where each search step can do one of two things:

1. Assign a new atom `a_* = 1`
2. Add a new inference cluster (for example locked candidates, hidden pairs, naked pairs)

After each action, TNReason propagates consequences via `propagate_until_convergence`.


## Main idea

The important design choice is:

- MCTS does not search over raw partial assignments
- MCTS searches over **closed states**

A closed state is a state after:

1. evidence is applied
2. active inference clusters are added to the propagator
3. message passing is run
4. forced atoms are extracted from the mean parameters
5. contradiction / solved / legal next actions are computed

This means each tree node already contains the current TNReason inference result.


## Files

### `actions.py`

Defines the action types:

- `AssignAction(atom_key)`
- `AddClusterAction(cluster_key)`
- `ClusterCandidate`


### `state.py`

Defines `AlphaSudokuState`.

This is the main state object used by the environment and MCTS.

It stores:

- `evidence`
- `active_inference_clusters`
- `assignment_actions`
- `cluster_candidates`
- `atom_priors`
- `contradiction`
- `solved`
- `value_estimate`
- `assigned_count`
- `message_count`


### `closure_engine.py`

Defines `AlphaSudokuClosureEngine`.

This is the deterministic TNReason solver core.

Responsibilities:

- build a forward message passer from `demonstrations/sudoku/sudoku_forward_mp.py`
- inject currently active inference clusters
- run `propagate_until_convergence`
- extract forced atoms from mean parameters
- validate consistency using `experiments/backtracking_search/check_caNetwork.py`
- generate legal assignment actions from `canParamDict`
- generate legal cluster-addition actions from rule detectors in `demonstrations/sudoku/inferenceClusters/`
- compute a heuristic value estimate

It also includes an internal closure cache so repeated MCTS visits to the same state do not recompute propagation every time.


### `env.py`

Defines `AlphaSudokuEnv`.

This is the AlphaZero-style environment wrapper.

Main methods:

- `reset(evidence)`
- `legal_actions(state)`
- `step(state, action)`

`step(...)` applies one action and then calls the closure engine again.


### `policies.py`

Defines `HeuristicPolicyValue`.

This is a placeholder policy/value component.

Right now it:

- prefers assignment actions slightly more than cluster actions
- uses TNReason-derived atom probabilities as assignment priors
- uses the state’s heuristic `value_estimate`

Later this should be replaceable by a trainable model.


### `mcts.py`

Defines `AlphaSudokuMCTS`.

This is a single-player PUCT search.

It:

- expands states using `AlphaSudokuEnv`
- uses priors from `HeuristicPolicyValue`
- backs up scalar values through the tree
- chooses the most visited root action


### `run_experiment.py`

Minimal entrypoint for running the current system on a benchmark puzzle.

It:

- loads a puzzle from `nikoli100.parquet`
- creates the closure engine, environment, and MCTS
- runs repeated search steps
- validates the final result by tensor contraction
- prints closure-cache stats


## Data flow

One search step looks like this:

1. MCTS receives the current `AlphaSudokuState`
2. MCTS asks the environment for legal actions
3. MCTS picks or explores one action
4. `env.step(...)` applies that action
5. `closure_engine.close(...)` runs TNReason propagation
6. A new `AlphaSudokuState` is produced
7. MCTS uses that result for further search and backup


## Action types

### 1. Assignment actions

Example:

```python
AssignAction("a_0_0_1_2_3")
```

This means:

- set that Sudoku atom to true

The closure engine then propagates all consequences.


### 2. Cluster actions

Example:

```python
AddClusterAction("a_..._row_..._square_...")
```

This means:

- activate a new inference cluster in the message-passing graph

These cluster candidates are generated from the rule detectors in:

- `demonstrations/sudoku/inferenceClusters/locked_candidates.py`
- `demonstrations/sudoku/inferenceClusters/hidden_pairs.py`
- `demonstrations/sudoku/inferenceClusters/naked_pairs.py`


## Why this architecture

This structure is meant for the later case where agents may become trainable.

It gives a clean split between:

- solver/inference code
- search code
- policy/value code

That matters because later you may want agents that:

- choose new assignments
- choose which inference clusters to activate
- work with hard features now
- possibly work with hybrid/soft features later

The closure engine isolates TNReason-specific logic so those later changes stay local.


## Cache

`AlphaSudokuClosureEngine` caches closure results for:

- `(evidence, active_inference_clusters)`

This is important because MCTS often revisits the same state.

Without this cache, the engine would repeatedly rebuild the propagator and rerun message passing for identical states.


## Current limitations

This is an architectural starting point, not a finished solver.

Known limitations:

- value function is still heuristic
- policy is still heuristic
- MCTS tree does not yet use a transposition table
- full runtime can still be slow because TNReason propagation is expensive
- the system currently uses the existing hard-feature forward message passing


## Run

From the repository root:

```bash
cd tnreason-py-version2\(2\)/tnreason-py-version2
python -m experiments.alphasudoku.run_experiment
```


## Suggested next steps

1. Add a transposition table to `mcts.py`
2. Add a learned policy/value module behind the `HeuristicPolicyValue` interface
3. Improve terminal validation and reward shaping
4. Experiment with hybrid features / modified propagation in the closure engine
