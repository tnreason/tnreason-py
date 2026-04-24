# Coding Instructions: Refactor `experiments/alphareason` into a contraction-only forward-chaining + DFS solver for Sudoku

## Goal

Replace the current **CANetwork/message-passing + MCTS** control flow in `experiments/alphareason` with a **contraction-network-only** solver that uses:

1. **Tier A**: existing forward chaining
2. **Tier B**: automatic discovery of small local contraction clusters that shrink supports
3. **Tier C**: bounded exact local contraction around an anchor bottleneck
4. **Tier D**: recursive DFS branch-and-propagate

Do **not** reintroduce Computation-Activation Networks for this solver. The new exact Sudoku path should operate purely on **constraint / contraction networks**.

The implementation should reuse as much of `alphareason` as is reasonable, but the primary control loop should become **DFS**, not MCTS.

---

## High-level design decisions

### Keep and reuse

Reuse these existing concepts where possible:

- `AlphaReasonTask`
- `AlphaReasonState`
- `AssignValueAction`
- closure cache in `AlphaReasonClosureEngine`
- Sudoku task builder and evidence/assignment mapping logic
- existing forward chaining backend (`GenericForwardChaining`)
- any existing constraint-network Sudoku construction code

### Stop depending on for the new solver path

Do not use these as core dependencies for the new solver path:

- `close_canetwork`
- message passing inferers
- rule detectors based on CANetwork message passing
- MCTS as the main exact solver

MCTS may remain in the folder for legacy experiments, but it should not be part of the main exact Sudoku solver path.

---

## Short answer on folder structure

It is **not too messy** to keep this work inside `experiments/alphareason`. A lot can be reused.

Recommended approach:

- keep `alphareason` as the general search/closure package
- add new contraction-only modules rather than overloading the old message-passing logic further
- preserve current APIs where easy, but do not force the new solver into the old MCTS assumptions

### Recommended file strategy

Keep existing files, but add or refactor toward:

- `closure_engine.py` — extend/refactor for contraction-only closure and cluster evaluation
- `state.py` — extend state with support sets and richer cluster metadata
- `actions.py` — add action types for automatic cluster activation and bounded local exact contraction
- `task.py` — extend task config for contraction-only cluster proposal heuristics
- **new:** `dfs_solver.py` — exact recursive branch-and-propagate solver
- **new:** `cluster_proposals.py` — automatic proposal of small local contraction clusters
- **new:** `bounded_local_contraction.py` — Tier C exact anchor contraction
- **new:** `support_analysis.py` — shared utilities for extracting supports / support shrinkage / singleton detection
- `tasks/sudoku.py` — switch the default exact Sudoku path to contraction-only closure
- tests — add focused tests for Tier A/B/C/D behavior

This keeps the folder coherent and allows reuse of task/state/action abstractions.

---

## Algorithm to implement

# Tier A — base forward chaining

Use the existing forward chaining implementation as the base closure operator.

### Required behavior

Given:

- a constraint network from `task.network_factory(evidence)`
- current explicit evidence
- active inference clusters / active exact local contraction modules

the closure should:

1. run forward chaining
2. detect contradiction
3. summarize each target feature’s current support
4. promote singleton-supported target features into assignments
5. update evidence with those singleton assignments
6. repeat until stable or contradiction

### Important

The closure result must explicitly contain **supports per target feature**, not only priors.

Add a state field for something like:

- `feature_supports: Dict[str, Tuple[int, ...]]`

This should become a first-class object in the state.

`feature_priors` can still exist, but the exact solver should reason mainly from supports.

---

# Tier B — automatic micro-cluster discovery

## Intent

Do **not** implement named detectors like “hidden pair detector” as the main approach.

Instead, implement a **generic cluster proposal + evaluate + accept** pipeline that discovers useful local exact inferences automatically.

Hidden-pair-like effects should emerge as a consequence of local exact support shrinkage.

## Core concept

A **micro-cluster** is a small induced local contraction subnetwork chosen around a bottleneck region.

It is used to answer:

> If I contract this small local cluster exactly under current evidence and project back to certain variables, do any supports shrink?

If yes, the cluster is useful and can be activated as an inference step.

## New action type

Add a new action type in `actions.py`.

Recommended:

```python
@dataclass(frozen=True)
class AddMicroClusterAction:
    cluster_key: str
```

You may either keep `AddClusterAction` and reinterpret it generically, or replace it with a more explicit name. Prefer clarity.

## New state metadata

Add richer cluster candidate metadata to `state.py`.

Recommended dataclass:

```python
@dataclass(frozen=True)
class ClusterCandidate:
    key: str
    colors: Tuple[str, ...]
    projected_features: Tuple[str, ...]
    estimated_cost: float
    estimated_gain: float
    tier: str  # "micro" or "bounded_local"
```

## Proposal pipeline

Implement a new module `cluster_proposals.py`.

Provide a function approximately like:

```python
def propose_micro_clusters(task, state, chainer, max_candidates: int = 20) -> Tuple[ClusterCandidate, ...]:
    ...
```

### Proposal heuristic requirements

Only propose micro-clusters around **bottlenecks**. Do not enumerate the whole puzzle globally.

Anchors should be chosen from:

- the unresolved feature with smallest support size
- optionally a few other unresolved features tied for smallest support
- optionally a “digit in a unit” bottleneck if task metadata can expose this cleanly

### Cluster growth heuristic

For a chosen anchor feature, propose clusters by combining nearby constraints/variables:

- the anchor cell feature and its incident atom variables
- one Sudoku unit neighborhood at a time (row / column / box)
- small overlaps of two units through the anchor
- optional digit-specific neighborhoods near the anchor

**Do not hand-code hidden pair / naked pair / fish rules.**

The generator should operate graph-theoretically from local overlap and budget limits.

### Budget limits for Tier B

Keep these clusters small.

Suggested defaults:

- unresolved target features touched: <= 4 or <= 6
- total local variables/colors: <= 12 or <= 16
- total local constraints/cores: <= 12 or <= 20

These numbers may be tuned, but Tier B must remain cheap.

## Cluster evaluation

Implement exact evaluation of a proposed micro-cluster.

Provide a function approximately like:

```python
def evaluate_cluster_gain(task, state, cluster_candidate, chainer) -> ClusterEvaluation:
    ...
```

where the evaluation computes:

- projected supports before
- projected supports after exact local contraction
- number of removed support values
- number of new singleton features
- contradiction flag
- a numeric gain score

### Gain score

Use a simple score such as:

```text
removed_supports
+ 3 * new_singletons
+ 10 * contradiction_bonus
- lambda * estimated_cost
```

Exact constants can be adjusted; keep it simple and deterministic.

## Acceptance rule

A micro-cluster should become a legal action only if it has **strictly positive exact gain** or clears a contradiction.

After accepting a micro-cluster action, the closure engine should treat it as an active local inference module and use it during closure.

### Important operational requirement

Tier B clusters are not one-off suggestions only. Once activated, they should be part of the closure process for descendant states.

In other words:

- `state.active_inference_clusters` persists down the search tree
- closure applies all active clusters after/base-forward-chaining until no more support shrinkage occurs

---

# Tier C — bounded exact local contraction

## Intent

Tier C is still a contraction cluster, but it is used differently.

Tier B = many cheap local contraction opportunities.

Tier C = occasional, more expensive **exact local subproblem solve** around one anchor bottleneck when Tier B stalls.

## New action type

Add another action type, e.g.:

```python
@dataclass(frozen=True)
class AddBoundedLocalContractionAction:
    cluster_key: str
```

If preferred, you can unify both Tier B and Tier C under one generic cluster action class and distinguish by candidate metadata (`tier="micro"` vs `tier="bounded_local"`).

## New module

Create `bounded_local_contraction.py`.

Provide something approximately like:

```python
def propose_bounded_local_clusters(task, state, chainer, max_candidates: int = 5) -> Tuple[ClusterCandidate, ...]:
    ...
```

and

```python
def run_bounded_local_contraction(task, state, cluster_candidate, chainer) -> ClusterEvaluation:
    ...
```

## Proposal policy

Only use Tier C when:

- state is not solved
- no contradiction
- Tier A + currently active clusters produce no further shrinkage
- no accepted Tier B candidate with sufficient gain is available

## Anchor selection

Choose one anchor bottleneck feature:

- smallest support size > 1
- break ties by neighborhood pressure / peer count / lower entropy if convenient

## Cluster construction

Build a larger local induced subproblem around the anchor.

For Sudoku this should include nearby constraints from:

- anchor cell
- anchor row
- anchor column
- anchor box
- possibly a second ring of overlap, but stay under budget

## Tier C budget

Suggested defaults:

- unresolved target features included: <= 10 to 20
- total local variables/colors: <= 30 to 50
- total local constraints/cores: keep bounded by an estimated contraction-cost threshold

The goal is still much cheaper than whole-network contraction.

## Output of bounded local contraction

Project exact local feasibility back to:

- the anchor feature at minimum
- optionally the anchor’s immediate unresolved peers

This action is useful if it yields:

- support shrinkage
- singleton creation
- contradiction

Do not require it to solve the whole local region.

---

# Tier D — DFS branch-and-propagate

## Core control loop

Create a new module `dfs_solver.py`.

Implement an exact recursive solver.

Recommended interface:

```python
class AlphaReasonDFSSolver:
    def __init__(self, closure_engine, max_micro_cluster_actions=3, max_bounded_local_actions=1):
        ...

    def solve(self, task):
        ...
```

## Required behavior

Given a state:

1. close it fully with Tier A plus all active clusters
2. if contradiction: fail / backtrack
3. if solved: return solution
4. if good Tier B micro-cluster actions exist: try a small number of them first
5. else if good Tier C bounded-local actions exist: try them
6. else branch on a target feature assignment

### Branching variable selection

Branch on the unresolved feature with smallest support size > 1.

Tie-break by one or more cheap heuristics:

- higher peer pressure / more unresolved neighbors
- lower entropy of current prior
- deterministic lexical key fallback

### Value ordering

Order values by:

- current `feature_priors[feature_key][value_index]` descending, if available
- otherwise deterministic ascending index

### Search strategy

Use depth-first search with backtracking.

Do **not** use MCTS for the main exact solver.

## Caching

Keep and extend closure caching.

Add optional search memoization on `state.state_key()`.

### Required key contents

At minimum, the key should include:

- explicit evidence
- active inference clusters
- active bounded-local clusters

If you unify cluster storage, one cluster dict is enough.

---

## Required refactors by file

### `state.py`

Extend `AlphaReasonState` to include:

- `feature_supports: Dict[str, Tuple[int, ...]]`
- richer `cluster_candidates`
- possibly separate candidate lists for micro vs bounded local

Recommended additions:

```python
feature_supports: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
micro_cluster_candidates: Tuple[ClusterCandidate, ...] = field(default_factory=tuple)
bounded_local_candidates: Tuple[ClusterCandidate, ...] = field(default_factory=tuple)
```

`legal_actions` may still be maintained for compatibility, but the DFS solver can also directly consume the separate candidate lists.

### `actions.py`

Add new action types or refactor `AddClusterAction` into a clearer hierarchy.

Recommended union:

```python
AlphaReasonAction = Union[
    AssignValueAction,
    AddMicroClusterAction,
    AddBoundedLocalContractionAction,
]
```

If minimizing churn is important, reusing one generic cluster action is acceptable, but action metadata must still distinguish tiers.

### `closure_engine.py`

This is the main refactor.

Implement or refactor toward:

- exact support extraction as a first-class result
- application of active micro-clusters during closure
- application of active bounded-local contractions during closure
- generation of new cluster candidates after closure stabilizes

#### Important

`close_constraint_network` should become the main closure path for Sudoku.

The closure pipeline should look like:

1. create `GenericForwardChaining(task.network_factory(evidence))`
2. run base forward chaining
3. summarize supports
4. apply active clusters that provably shrink supports
5. if supports shrink, feed resulting singleton assignments back into evidence and repeat
6. stop when fixed point or contradiction
7. propose fresh Tier B and Tier C candidates from the stabilized state
8. build `AlphaReasonState`

### `task.py`

Extend task configuration to support cluster proposal heuristics.

Suggested new optional fields:

```python
micro_cluster_proposer: Optional[Callable[..., Tuple[ClusterCandidate, ...]]] = None
bounded_local_proposer: Optional[Callable[..., Tuple[ClusterCandidate, ...]]] = None
cluster_budget_config: Dict[str, object] = field(default_factory=dict)
```

Do not keep `rule_detectors` as the main path for the new exact Sudoku solver.

### `tasks/sudoku.py`

Refactor the constraint Sudoku builder so the default exact path uses the new contraction-only closure engine.

For Sudoku tasks:

- expose enough metadata for local neighborhood proposal heuristics
- keep assignment-evidence mapping as-is
- keep target feature keys as cell features
- optionally add metadata helpers for row/col/box peers if this simplifies cluster proposal

### `mcts.py`

Do not delete unless necessary, but do not make it part of the main exact solver path.

### `env.py`, `policies.py`

Only refactor if needed for compatibility; not central to the new exact solver.

---

## Concrete closure behavior to implement

The closure engine should now distinguish clearly between:

- **explicit evidence**: assignments already made
- **derived singleton assignments**: discovered by exact support reduction
- **active clusters**: local exact reasoning modules available during closure
- **candidate clusters**: local exact reasoning modules worth trying next in search

### Minimal closure pseudocode

```python
close_constraint_network(task, evidence, active_clusters):
    contradiction = False
    working_evidence = dict(evidence)

    while True:
        changed = False

        chainer = GenericForwardChaining(task.network_factory(working_evidence))
        run_base_forward_chaining(chainer)
        if contradiction_detected:
            return contradiction_state

        supports = summarize_all_target_supports(chainer)
        if any empty support:
            return contradiction_state

        singleton_assignments = extract_singletons_not_already_in_evidence(supports)
        if singleton_assignments:
            working_evidence.update(singleton_assignments_as_evidence)
            changed = True
            continue

        cluster_shrinkage = apply_active_clusters_exactly(task, working_evidence, chainer, active_clusters)
        if cluster_shrinkage.causes_contradiction:
            return contradiction_state
        if cluster_shrinkage.new_singletons_or_support_shrinkage:
            working_evidence.update(cluster_shrinkage.singleton_evidence)
            changed = True
            continue

        if not changed:
            break

    state = build_state_from_stable_supports(...)
    state.micro_cluster_candidates = propose_micro_clusters(...)
    state.bounded_local_candidates = propose_bounded_local_clusters(...)
    return state
```

### Important implementation note

If exact support shrinkage from an active cluster does not directly create a singleton, the state must still retain the tightened supports somehow.

Preferred approach:

- represent active clusters as actual active inference modules whose projected support restrictions are re-applied during closure

Avoid a design where non-singleton support reductions are computed and then discarded.

A practical representation is to let active clusters contribute extra local constraint cores or equivalent restricted summary cores to the closure network. If that is too invasive initially, store cluster-derived support restrictions per feature and reapply them during closure.

---

## Practical simplification allowed

A good first implementation may treat an accepted cluster as a **persistent support-restriction module** rather than as an explicit reusable TN core, as long as:

- it is exact
- it persists in descendant states
- it narrows supports correctly
- closure reaches a correct fixed point

This is acceptable if integrating cluster restrictions directly back into the network is cumbersome.

---

## Tests to add

Create or extend tests under `experiments/alphareason/tests`.

### Tier A tests

- forward chaining computes singleton supports correctly on small Sudoku instances
- contradiction is detected when a target feature has empty support

### Tier B tests

- micro-cluster proposal returns local candidates only near bottlenecks
- evaluation of a micro-cluster can shrink supports on a known small Sudoku pattern
- accepted micro-cluster persists in descendant state and is reused during closure

### Tier C tests

- bounded local contraction proposes only a few anchor-centered candidates
- bounded local contraction yields support shrinkage or contradiction on a crafted hard local situation

### Tier D tests

- DFS solver solves easy Sudoku without branching explosion
- DFS solver backtracks correctly on contradiction
- DFS solver prefers cluster actions before assignment branching when gain is positive

### Regression tests

- existing non-Sudoku alphareason tests should not be broken unnecessarily
- if some old CANetwork-specific tests no longer apply to the main path, isolate them rather than forcing the new code into their assumptions

---

## Coding priorities

Implement in this order:

1. extend `AlphaReasonState` with support sets and richer candidate metadata
2. refactor `close_constraint_network` to make supports first-class and stable
3. implement `dfs_solver.py` with branch-and-propagate using current supports
4. implement `cluster_proposals.py` with automatic micro-cluster proposal around MRV bottlenecks
5. implement exact cluster evaluation and persistent support restriction mechanism
6. implement `bounded_local_contraction.py`
7. add/repair tests

Do not start with elaborate cluster heuristics before the DFS backbone is working.

---

## Non-goals

Do not do these in the first implementation:

- whole-network contraction for Sudoku
- complicated learned policies
- MCTS-driven exact solving
- large-scale hand-coded Sudoku pattern zoo
- over-generalizing before Sudoku works well

---

## Acceptance criteria

The implementation is successful if:

1. the exact Sudoku path runs on constraint/contraction networks only
2. Tier A forward chaining remains functional
3. Tier B proposes small local contraction clusters automatically without named hidden-pair detectors
4. Tier C performs occasional larger bounded exact local contraction around a bottleneck
5. Tier D DFS search solves puzzles by branch-and-propagate
6. existing `alphareason` abstractions are reused where sensible
7. the code remains reasonably organized inside `experiments/alphareason`

---

## Final guidance for the coding agent

Prefer a **clean, incremental refactor** over a giant rewrite.

It is acceptable to keep legacy CANetwork/MCTS code in the folder, but the new exact Sudoku path should be clearly separated and easy to follow.

Aim for this conceptual architecture:

- `closure_engine.py` = exact local consequence extraction
- `cluster_proposals.py` = automatic proposal of local contraction opportunities
- `bounded_local_contraction.py` = selective stronger local exact reasoning
- `dfs_solver.py` = exact search controller
- `tasks/sudoku.py` = Sudoku-specific task metadata and network construction

The key idea is:

> use contraction-based local exact reasoning as far as possible, and use DFS only when local exact reasoning stalls.
