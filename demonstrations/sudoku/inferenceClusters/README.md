# Standalone Sudoku inference clusters

This library selects small collections of original Sudoku tensors and computes
their exact Boolean support. Selection, description, contraction, and application
are separate operations. Importing the package runs no detectors and changes no
solver or benchmark settings. The researched rule catalogue, construction details,
and deliberately unsupported cases are in [COVERAGE.md](COVERAGE.md).

The new interface is independent of MAC. Existing `naked_pairs.py`,
`hidden_pairs.py`, `locked_candidates.py`, and the explicit helpers in `mac.py`
remain available for existing callers.

## Encoding and support

Write `n = num**2`: `num=2` means 4x4 and `num=3` means 9x9. Digits, rows,
columns, and domain values are zero based. A cell at `(row, col)` has category
`pos_{row//num}_{row%num}_{col//num}_{col%num}`. Its digit indicator appends the
digit to the same coordinates, with prefix `a_` instead of `pos_`.

Each cell category chooses a digit. A row/column/box category chooses the
position of one particular digit; its integer domain values are indices in
`get_sudoku_constraints(num)[category]`, **not digit values**. `ctx.pos`,
`ctx.atom`, and `ctx.unit` construct these names without changing the established
encoding.

An original binary link relates a category to one Boolean indicator. For the
standard encoding its support says that the indicator is true exactly when the
category chooses that indicator's index. Descriptions select original tensors
by their scopes, so tensor keys need not follow a particular naming convention.
All original tensors having a selected scope are included. Missing required
links and incompatible dimensions are rejected.

Contraction evaluates each original tensor's **realized value != 0**, then uses
Boolean AND and existential OR. Negative weights are supported; arithmetic
cancellation between different tensors cannot destroy a valid assignment.
Overlapping sparse slices are evaluated before deciding support, so cancellation
inside an individual tensor is respected. Tensor iterators are not consumed.
Bounded readers support NumPy-backed cores, `PolynomialCore`, and `PandasCore`;
other non-NumPy backends require a bounded support-reader adapter and are
explicitly rejected. Sparse work includes comparisons against stored rows or
slices, including cancelled entries, multiplied by their scope rank.
Detectors use the standard Sudoku geometry as a selection heuristic even when
the supplied link supports differ; only the exact contraction determines any
reported elimination.

## Standalone use

Run these examples from the repository root with the package available on the
Python path. This example constructs constraints directly, without a MAC state:

```python
from tnreason.application import create_categorical_cores
from demonstrations.sudoku.constraints.sudoku_constraints import get_sudoku_constraints
from demonstrations.sudoku.inferenceClusters.core import RuleContext, contract_cluster
from demonstrations.sudoku.inferenceClusters.subsets_fish import detect_locked

cores = create_categorical_cores(get_sudoku_constraints(num=2))
# Digit 0 in the top-left box is confined to its first row.
ctx = RuleContext(cores, {"square_0_0_0": (0, 1)}, num=2)
selection = detect_locked(ctx)
cluster = selection.clusters[0]
result = contract_cluster(ctx, cluster)
assert result.status == "executed"
assert result.supported_domains["row_0_0_0"] == (0, 1)
```

`RuleContext` also accepts a `ConstraintNetwork`. It snapshots and normalizes
domains, treating unspecified variables as unrestricted. It does not propagate
them. For more complete detection, an external caller can supply domains after
ordinary GAC. Construct a fresh context after any domain change, including a
change to a sibling branch. Neither input domains nor original tensors are
modified.

Each `RuleCluster` wraps the existing `SummaryClusterCandidate` as `.candidate`;
it also exposes `.core_keys`, `.open_colors`, `.participating_variables`,
`.internal_variables`, `.domain_inputs`, `.assumptions`, and `.evidence`.
`ctx.describe(family, categories, atoms=None, outputs=None, evidence=())` can
describe caller-chosen motifs without running a detector. By default it selects
the indicators shared by at least two chosen categories and retains all chosen
categories. Explicit `atoms` and `outputs` allow smaller, targeted projections.

Every detector returns `Detection(clusters, steps, truncated)`. A recognized
pattern is not a promised elimination: contraction may find no extra restriction
or may return `status="skipped_budget"`. An executed result contains Boolean
`summaries`, `supported_domains`, a `contradiction` flag, and resource accounting.
A skipped result has empty summaries and a reason; it must not be treated as
evidence that the cluster is satisfiable. Malformed inputs and unencoded extra
assumptions raise errors.

The default contracts a separate exact unary projection for each retained
output. This keeps allocation manageable and reports precisely the supported
values of those outputs. `joint=True` retains their joint relation when it fits
the budget; its correlations can strengthen subsequent propagation further.
Internal variables are existentially eliminated in both modes. Domains of every
participating variable enter the default computation. Constraints and domains
outside the selected cluster do not enter it; thus local support can retain
values a larger cluster would exclude, but cannot exclude a satisfying global
assignment under the supplied domains.

## Explicit family selection

The legacy selectors remain available. The expanded selectors share a few
parameterized constructions; named calls restrict the corresponding geometry.
All calls below are also exported from this package.

| Module | Explicit selectors and scope |
| --- | --- |
| `subsets_fish.py` | `detect_singles`, `detect_locked`, `detect_subsets(hidden=..., sizes=...)`, `detect_fish(kind="basic"/"finned"/"sashimi", sizes=...)`. |
| `advanced.py` | Existing `detect_xy_wings`, `detect_xyz_wings`, `detect_xy_chains`, `detect_simple_coloring`, and disjoint `detect_als_xz`. |
| `implication.py` | `detect_aic`, `detect_x_chains`, `detect_grouped_aic`, `detect_nice_loops`; `detect_turbot_fish`, `detect_skyscrapers`, `detect_two_string_kites`, `detect_empty_rectangles`, `detect_w_wings`; `detect_multi_coloring`, `detect_medusa`. |
| `fish_general.py` | `detect_general_fish`, `detect_franken_fish`, `detect_mutant_fish`, `detect_multi_fin_fish`, `detect_endo_fish`, `detect_cannibalistic_fish`, `detect_siamese_fish`. Optional explicit `bases`, `covers`, or `cover_pairs` select a concrete motif without an exhaustive prefix scan. |
| `als_extended.py` | `detect_als_xz_extended(kind="singly"/"doubly"/"both", allow_overlap=...)`, `detect_als_chains`, `detect_als_xy_wing`, `detect_death_blossom`, `detect_bent_subsets`, `detect_wxyz_wings`, `detect_sue_de_coq`, `detect_aligned_exclusion(size=2/3/...)`. |
| `forcing.py` | Finite exact-one proof tracing for forcing chains/nets; original-tensor compound Kraken proofs. See the coverage table for type-specific conditions. |
| `projections.py` | `detect_templates`, `detect_pattern_overlay`, `detect_nishio`, `detect_proof_regions`, `detect_complete_support`. Full proof regions and complete support are general projection proposals, rather than assertions that a named human pattern was detected. |

The registry requires an explicit finite family selection and returns a mapping
from family name to `Detection`. `available_rules()` lists the actual accepted
names. Aliases and parameter restrictions are not counted as independent proof
mechanisms. There is no default selection and no detector runs on import.

```python
from demonstrations.sudoku.inferenceClusters import available_rules, detect_rules

chosen = detect_rules(ctx, families=("locked_candidates", "basic_fish", "aic"),
                      options={"basic_fish": {"sizes": (2,)}})
fish = chosen["basic_fish"]
```

`RuleRegistry(families).detect(ctx, options=...)` provides repeated explicit use.
Unknown or assumption-gated names raise `UnsupportedRuleError`. Options for an
unselected family are rejected. Each selected family receives its own scan
budget: `F` families allow at most `F * max_steps` charged scan steps and
`F * max_patterns` returned descriptions. Truncation is reported per family.

Automatic discovery is deliberately finite. When a known large motif lies
outside an automatic prefix, callers can use explicit house selections, validated
`make_als`/`describe_als_*`/`describe_death_blossom`/`describe_sue_de_coq`/
`describe_aligned_exclusion`, or `describe_implication_paths`. These helpers
still contract only original tensors. ALS/counting descriptors validate their
pattern predicates. Low-level path and compound descriptors validate finite
shape, original provenance, and required case coverage; they do not accept a
caller's implication labels as proof of an elimination. Exact contraction of
the original premises determines whether any claimed consequence follows.
`combine_clusters` and `ctx.describe_tensors(family, original_keys, outputs)`
express arbitrary finite proof unions, including higher-order original tensors.
A proof description is not a cached contraction result.

For example, a template description joins all row/column/box constraints for
one digit; a multi-digit overlay additionally couples them through the cell
constraints. These calls perform no template or completed-grid enumeration:

```python
from demonstrations.sudoku.inferenceClusters import (
    describe_templates, describe_pattern_overlay, contract_cluster,
)

template = describe_templates(ctx, 0)
template_result = contract_cluster(ctx, template)
overlay = describe_pattern_overlay(ctx, (0, 1), outputs=(ctx.unit("row", 0, 0),))
overlay_result = contract_cluster(ctx, overlay)  # inspect status before using it
```

An ALS chain can also be described directly. Here its middle node consists of
two row cells with three digits; the endpoint digit is unsupported at `(3,3)`:

```python
from demonstrations.sudoku.inferenceClusters import describe_als_chain

als_ctx = RuleContext(cores, {
    "pos_0_0_0_0": (0, 3), "pos_0_0_0_1": (1, 3),
    "pos_0_0_1_1": (0, 2), "pos_1_1_0_1": (1, 2),
}, num=2)
als_cluster = describe_als_chain(
    als_ctx, (((0, 3),), ((0, 0), (0, 1)), ((3, 1),)),
    (0, 1), 2, (3, 3),
)
als_result = contract_cluster(als_ctx, als_cluster)
assert als_result.status == "executed"
assert als_result.supported_domains[als_ctx.pos(3, 3)] == (0, 1, 3)
```

`describe_complete_support(ctx, outputs=...)` names the entire original network.
In principle every deduction entailed by those finite encoded constraints is a
projection of that relation. This does not imply affordable contraction or
complete discovery by the named local selectors. A large full-network proposal
can exceed `max_cores`, `max_work`, or `max_size`, and is reported as skipped.
Explicit additional variant constraints participate when included in the
original network; no diagonal, cage, anti-knight, or uniqueness axiom is invented.

## Bounded detection and execution

`RuleLimits` makes the following defaults explicit. They apply to the new
standalone interface; existing legacy helpers retain their existing behavior.

| Limit | Default | Meaning |
| --- | ---: | --- |
| `max_board_size` | 16 | Maximum `num**2`. |
| `max_network_cores` | 100,000 | Maximum original input tensors inspected. |
| `max_domain_values` | 1,000,000 | Cap on declared dimension sum; separately on supplied domain entries read. |
| `max_steps` | 100,000 | Detector trials or path expansions; reported as `steps`. |
| `max_patterns` | 64 | Returned descriptions per scan. |
| `max_subset_size` | 4 | Maximum selectable subset cardinality. |
| `max_fish_size` | 4 | Maximum number of base/cover sets. |
| `max_chain_length` | 8 | Cells in legacy XY/conjugate paths; edges in implication paths; ALS nodes in ALS chains; forcing trace depth. |
| `max_als_size` | 3 | Maximum cells in each ALS/near-locked helper or SDC side. |
| `max_als_sets` | 128 | Intermediate catalogue entries for ALS/counting scans. |
| `max_petals` | 4 | Death Blossom petals, forcing source alternatives, or aligned helpers. |
| `max_graph_nodes` | 2,048 | Implication/group nodes; also forcing active-indicator and per-branch literal caps. |
| `max_graph_edges` | 16,384 | Stored strong/weak graph edges. |
| `max_group_size` | 3 | Indicators in each grouped implication node. |
| `max_coloring_nodes` | 32 | Nodes per conjugate coloring component. |
| `max_net_categories` | 24 | Original category premises in automatic forcing traces/regions; explicit compound descriptions use their own proof-count limits and execution budgets. |
| `max_net_radius` | 3 | Maximum neighborhood radius for structural proof regions. |
| `max_cores` | 256 | Maximum selected original tensors per contraction. |
| `max_work` | 10,000,000 | Dense support reads, sparse slice/row comparisons, and Boolean join/reduction entries. |
| `max_planning_steps` | 10,000,000 | Cumulative factor-scope examinations while planning and selecting eliminations. |
| `max_size` | 1,000,000 | Entries in any individual input/intermediate tensor. |
| `max_output_size` | 100,000 | Entries in each expanded output tensor. |
| `max_cache_entries` | 128 | Cached original-support arrays; zero disables caching. |

For example, pass `limits=RuleLimits(max_steps=5000, max_patterns=8,
max_work=1_000_000)` when constructing a context. Raising a size parameter also
requires raising its corresponding configured limit. Hitting a scan limit sets
`truncated=True`; it says nothing about patterns beyond the scanned prefix.

The executor compacts axes to current domains, squeezes fixed axes, and combines
scalar factors. It eliminates local degree-one/two Boolean indicators early.
Higher-degree indicators and categorical variables use a verified greedy bucket
path that chooses the smallest joined scope next. It checks allocations before each
join and output expansion; an intermediate above rank 31 is explicitly skipped.
It does not allocate one monolithic indicator cube or perform a search for a
completed solution. Work is cumulative across the output projections of one
execution. `max_size` bounds an individual tensor, rather than total resident
memory; original arrays, factor lists, and bounded cache entries also occupy
memory. A large pattern can be recognized successfully and still be skipped by
the executor. `ContractionResult.work` reports charged tensor-entry work;
`.planning_steps` separately reports charged factor-scope examinations. The
reported `.max_size` is the largest considered allocation and can include a
rejected size on a skipped result; allocation checks precede array creation.
Detector `steps` counts structural trials/edge examinations, not CPU
instructions. An option query inspects up to `n**2` local entries; a trial's
finite inner work also depends on the explicit board, pattern, group, and
catalogue bounds. None of these selectors hides an unbounded assignment search.
Graph/ALS catalogue caps mark detection truncated; a configured pattern length
excludes longer structures even when the scan itself finishes.

## Domains, caching, and opt-in MAC use

Original support arrays have a context-local LRU cache keyed by tensor identity
key, scope, shape, and realized support bytes. Changing original values forces
a fresh support fingerprint. No domain-dependent contracted summary is cached.
Contexts keep references to original cores; do not mutate cores concurrently
with an execution or change scopes/dimensions after constructing a context.

`contract_cluster(..., use_domains=False)` omits all domain indicators. Such a
summary is valid under the matching original constraints for any domains. A
caller caching these summaries must include original realized supports, scopes,
selected keys, outputs, and assumptions in its cache identity. The default
`use_domains=True` result is valid only within its exact domain restriction and
descendants. Never reuse it in a sibling branch or another puzzle.

An external caller may explicitly use the generic MAC `add_cores` interface.
For example, a single root round with a caller-selected rule list is:

```python
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from demonstrations.sudoku.inferenceClusters import detect_rules
from demonstrations.sudoku.inferenceClusters.core import RuleContext, contract_cluster

network = ConstraintNetwork(cores)
state = network.create_mac_state({"square_0_0_0": (0, 1)})
families = ("locked_candidates", "basic_fish")

def selected_round(state, tag):
    if not state.propagate():
        return
    ctx = RuleContext(network, state.domains, num=2)
    selections = detect_rules(ctx, families=families)
    clusters = [cluster for selection in selections.values()
                for cluster in selection.clusters]
    for index, cluster in enumerate(clusters):
        result = contract_cluster(ctx, cluster)
        if result.status == "executed":
            # A fresh tag per persistent round avoids reusing an added core key.
            additions = {f"{tag}_{index}_{key}": core
                         for key, core in result.summaries.items()}
            state.add_cores(additions, reversible=True)
    state.propagate()

selected_round(state, "root_round_0")
solution = network.solve_mac(state=state)
```

This example performs one requested root round; ordinary MAC then proceeds
without further Sudoku detection. `state.domains` supplies the exact snapshot
used for every result in that round, so subsequent shrinking remains sound.
At a non-root decision level, additions **must** be reversible:

```python
before = state.snapshot_domains()
state.push_level()
try:
    state.restrict_domain("pos_0_0_0_0", (0, 1))
    selected_round(state, "trial_round_0")  # fresh context inside this branch
finally:
    state.pop_level()
assert state.snapshot_domains() == before
```

Create a level before adding summaries if they should later be removed.
`pop_level` restores both branch additions and domains. A caller deciding which
families to run, when to repeat a round, which outputs to retain, or whether to
install its own closure callback is choosing an experiment policy; the library
does not install one. Generic MAC has no Sudoku imports or detector defaults.

## Verification

Focused tests live in `demonstrations/sudoku/inferenceClusters/tests/`; existing
Sudoku pair/locked-cluster tests remain in `experiments/constraint_networks/tests/`.
They check independently specified
patterns and invalid near-patterns, exact support, transposes, contradictions,
input preservation, changed domains, budgets, and reversible application.
The coverage catalogue distinguishes tested implemented selectors from methods
for which only a potential translation is described.
