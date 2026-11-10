# Learning useful tensor clusters

Learn to construct clusters from original tensor cores and choose their open variables. Every proposed deduction is checked by domain-conditioned contraction and GAC.
The supervised implementation is separate and unchanged.

Generated results and trained checkpoints are preserved locally and excluded from
the source checkout. The `results/...` paths below identify historical artifacts.
To repeat commands that use them, restore the original archive or supply your own
data and checkpoints. Small seed datasets are in [`../data/`](../data/); training
and evaluation workflows write new outputs to the paths you provide.

On the completed matched-source 64-puzzle test, the older teacher with
`sampling_grow` uses **119 DFS nodes**, versus **201** for the structural heuristic
and **147** for matched random weights. The new teacher gives 137; new REINFORCE
sampling gives 156 versus matched random sampling's 154. The earlier easier-source
test showed no convincing learned edge. See the separate tables and audits below;
these results support useful learned construction while leaving its generalization
and further reinforcement training open.

## Entry points

Run Python entry points with `python -m experiments.cluster_selection.reinforcement.NAME --help`.

| Entry point | Purpose |
|---|---|
| [learn.py](learn.py) | Current teacher imitation and REINFORCE learner. Select the objective explicitly. |
| [qlearn.py](qlearn.py) | Constructive Q-learning with replay and a target network; pruning reward. |
| [experiment.py](experiment.py) | Generic propagation/DFS comparison with plain MAC and structural/random controls. |
| [ensemble_experiment.py](ensemble_experiment.py) | Frozen-checkpoint comparison using an archived reference protocol; interleave three learned constructors. |
| [scale_data.py](scale_data.py), [prepare.py](prepare.py) | Add independent unresolved Sudoku puzzles, or branch contexts from existing training puzzles. |
| [diagnose.py](diagnose.py) | Fixed-teacher development diagnostic comparing ordinary and repair imitation. |
| [train.py](train.py) | Legacy REINFORCE with a treewidth penalty. Also supplies shared state/checkpoint helpers used by current learners. |
| [submit_teacher_branches.sh](submit_teacher_branches.sh) | Prepare and optionally submit a shared-filesystem HTCondor warmup and three continuation branches. |

## Current findings and data preparation

The 6 November diagnostic (`results/constructive_improvement_20261106.903639e9/REPORT.md`)
found two concrete issues: depth-first inference spent its allowance on output
variants of almost one core set, and **64/64 sampled puzzles from the earlier
scaled training dataset have multiple valid solutions**. Discrepancy decoding improves validation DFS with
unchanged weights. Ambiguous puzzles can teach pruning, but sound cluster pruning
cannot reduce them to a single complete assignment without making a choice.

The fresh 64-puzzle unique evaluation (`results/unique_clusters_20261106.7eeea74f/REPORT.md`)
completed all solves: plain MAC 157 DFS nodes, heuristic 138, old learned decoder
140, learned discrepancy 131. The untrained discrepancy control achieved **116**
and more branch-free solves, using substantially more tensor work. Thus the
measured decoder gain does **not** demonstrate that learned weights beat random
construction under the equal contraction caps. These results must remain visible
when assessing the next training run.

`scale_data.py --unique-only --shards train_0.parquet` adds an optional data filter.
It verifies the reference against the original network, then uses a linear chain
of small generic tensor cores to require an alternative assignment. Only a complete
UNSAT proof certifies uniqueness; limits remain `unknown`. Source solutions are
preparation checks, never policy inputs. Existing files/defaults are preserved.
**This option certifies newly added rows only.** For an entirely certified training
set, start from a base with no legacy training rows; retain validation separately.
Per-row proof outcomes and limits are saved in preparation metadata.

The completed larger certified dataset (`results/unique_training_10k_20261106.f9b9837e/development.json`)
contains **10,000 training puzzles and 55 separate validation puzzles**. The
validation set consists of 23 earlier puzzles and 32 newly reserved puzzles;
all 10,000 training roots stall under GAC and have complete uniqueness proofs
over the 81 primary variables. There are no legacy training rows, and the original
network cores are identical. The independent audit (`results/unique_training_10k_20261106.f9b9837e/independent_review_complete.json`)
checks every source record and confirms no grid or unique-solution overlap with
the new validation or prospective test set. The earlier 1,024-puzzle preparation (`results/unique_training_20261106.d64151f6/completed.json`)
is preserved. A training-only restriction to at most 35 givens reduces trivial
preprocessing; actual GAC and uniqueness determine acceptance. Source solutions
and proof metadata are preparation checks, never policy inputs.

The clean-data pilot (`results/clean_pilot_20261106.a0to1e_c/independent_completed_review.json`)
used 128 training puzzles and the 32 new validation puzzles. Both learners started
from identical random weights and made 128 updates:

| Validation DFS nodes | Random initialization | Teacher final | REINFORCE final |
|---|---:|---:|---:|
| Discrepancy decoding | 119 | 139 | 141 |
| Sampling complete paths | 88 | 96 | **82** |

All solves completed with valid assignments. The REINFORCE sampling comparison (`results/clean_rl_sampling_20261106.5c12ca87/summary.json`)
has eight wins, 21 ties and three losses against matched random sampling.
The older teacher's unchanged weights score 79 with sampling on these same
validation puzzles. These are validation findings, not test-set learning claims.
The teacher pilot uses experimental greedy imitation; the REINFORCE pilot has
imitation weight zero. Its actual configuration is preserved in the pilot protocol.
Teacher training also shortened greedy clusters and reduced pruning on training
roots, so lower imitation loss alone is insufficient evidence of improvement.

Ranked validation correctly kept the initial weights as both pilots' `.best.pt`.
To reproduce the 82-node result, use the REINFORCE **final** `policy.pt` with
`--decoder sampling`; do not silently substitute its ranked-validation best.
REINFORCE without imitation gradients uses `learn.py --objective pruning --warmup 0
--imitation-weight 0 --decoder sampling`, with explicit construction/data budgets.
With multiple steps per puzzle, verified teachers can still advance the sampled
training state when none of the policy samples prunes. The one-step pilot has
no subsequent state that could receive that assistance.

The subsequent fresh test (`results/clean_pilot_test_20261106.95be3faf/summary.json`)
completed all 64 puzzles for each frozen method:

| Method | DFS nodes | Without branching |
|---|---:|---:|
| Plain MAC | 156 | 0/64 |
| Structural heuristic | 146 | 3/64 |
| Random weights, sampling | 88 | 41/64 |
| New REINFORCE final, sampling | 87 | 45/64 |
| New teacher final, sampling | 116 | 23/64 |
| Older teacher, sampling | 106 | 29/64 |

The one-node REINFORCE advantage over random is inconclusive (seven wins,
51 ties, six losses) and uses 30.6% more estimated tensor work. A separate
growth extension (`results/growth_heldout_20261106.lh7249cb/solve_summary.json`)
uses 74 nodes with random weights, 73 with the new teacher, and 75 with the older
teacher. Random and new teacher growth both solve 55/64 without branching.
Growth was frozen from validation after the original test started but before
comparative test totals were inspected; it did not change the original protocol.
These results support improved generic construction, not a convincing learned
advantage over matched random weights. All methods have the same 32-contraction
allowance and per-cluster bounds; actual tensor work differs.

The additional matched-source test (`results/matched_source_test_20261106._de5ty3e/summary.json`)
completed on 8 November. It uses 64 new unique, GAC-stalled puzzles drawn with the
training-data filters, with prior grid/solution overlaps excluded. It was reserved
after the earlier distribution difference was observed; it is a separate diagnostic,
and does not replace that earlier test. All seven unchanged methods were frozen
from validation before this batch was queried:

| Method | DFS nodes | Without branching |
|---|---:|---:|
| Plain MAC | 264 | 0/64 |
| Structural heuristic | 201 | 6/64 |
| Random weights, sampling | 154 | 17/64 |
| New REINFORCE final, sampling | 156 | 19/64 |
| Random weights, growth sampling | 147 | 24/64 |
| New teacher final, growth sampling | 137 | 26/64 |
| Older teacher, growth sampling | **119** | **33/64** |

All 448 solves are valid and uncensored; the
independent audit (`results/matched_source_test_20261106._de5ty3e/independent_results_review_20261108.json`)
verifies the raw totals and frozen hashes. Old-teacher growth wins 23 puzzles,
ties 40 and loses one against matched random growth, saving 19.0% of DFS nodes
and using 16.3% of its estimated tensor work. New-teacher growth saves 6.8%, with
14 wins and seven losses and 24.3% more work: a smaller, limited learning gain.
New REINFORCE fails to improve on its matched control. This identifies a useful
current constructor while keeping the role of earlier teacher training separate
from the new 128-update pilots. The full clean10k training job has not run here.

Ready-to-load growth exports (`results/growth_exports_20261106.khccwf_d/deployment_exports.json`)
preserve the older teacher and new pilot teacher weights, with the validated
`sampling_grow` setting recorded explicitly. They were chosen on validation;
their export does not represent a new training update.
The current validation-selected reference is
policy.parent_sampling_grow.pt (`results/growth_exports_20261106.khccwf_d/policy.parent_sampling_grow.pt`).

The prepared next GPU job (`results/reinforce_sampling_grow_gpu_20261106.q3mc4c0g/README.md`)
trains from random initialization on the full clean10k dataset using
`sampling_grow`, pruning reward and REINFORCE-only gradients. It plans ten passes
with teacher-assisted exploration of later states. This session has not submitted
it, and no submission/training artifacts were visible on 8 November; the remote
queue was not checked. The archived instructions include Condor parsing, submission,
monitoring and exact budgets. For this SSHFS shared checkout the files are already
in the shared home; the earlier copying prerequisite is unnecessary. This proposed
configuration has no GPU training result yet.

To reproduce the established three-branch workflow on the new data instead,
run from the shared repository on **bernoulli**:

```bash
bash experiments/cluster_selection/reinforcement/submit_teacher_branches.sh \
  /Home/optimier/schuette/.conda/envs/alphareason/bin/python \
  --data experiments/cluster_selection/reinforcement/results/unique_training_10k_20261106.f9b9837e/development.json \
  --passes 10 --chunk-size 500 --seeds 1 --max-jobs 3 \
  --teacher-candidates 256 --teacher-max-steps 65536 --nested-teacher \
  --cluster-attempts 32 --decoder discrepancy --validation-seconds 10800
```

This reproduces the established branching workflow, including imitation in the
REINFORCE branch; it is a different experiment from the pure-REINFORCE pilot.
Add `--dry-run` to inspect generated files first. This starts warmup from random
weights, then forks teacher continuation, REINFORCE plus imitation, and Q-learning
from its validation-selected warmup. It uses the established teacher loss;
the optional greedy loss below has not yet earned promotion. This command is
provided for the next GPU experiment; no new HTCondor job was submitted here.
With 10,000 unresolved pools, this plans 20 chunks per pass and 200 per phase.
Actual GAC-closed context deduplication determines the learner's reported pool count.

## Reproducing the earlier larger-data workflow

The scaled development data (`results/scaled_sudoku_20261028_seed29/development.json`) contain
10,000 new puzzles verified unresolved by the current MAC GAC on the unchanged tensor network.
Together with existing puzzles, there are **10,092 training indices, 10,051 unresolved training
roots and 23 validation puzzles**. Puzzles 85 and 91 moved into training and are excluded from validation.
A root is a puzzle's initial GAC closure; a context can also be a later domain state from that puzzle.
Equivalent closures are deduplicated, so the learner's logged pool/context counts determine actual coverage.

Preparation scanned 10,240 source rows; all 10,000 retained puzzles came from `train_1.parquet`.
That preparation did not use the larger `train_0.parquet`. Difficulty only orders the scan; current MAC GAC
controls inclusion. Source validation and reserved test grids are excluded. Stored solutions
verify data integrity and are never policy inputs. The earlier AlphaReason solved/unsolved
filter is not reused. See the preparation protocol (`results/scaled_sudoku_20261028_seed29/protocol.json`).

The following command reproduces that earlier corpus and run; it does not apply
the new uniqueness filter. From the shared repository on **bernoulli**, using the
shared CUDA-enabled environment:

```bash
bash experiments/cluster_selection/reinforcement/submit_teacher_branches.sh \
  /Home/optimier/schuette/.conda/envs/alphareason/bin/python \
  --data experiments/cluster_selection/reinforcement/results/scaled_sudoku_20261028_seed29/development.json \
  --passes 10 --chunk-size 500 --seeds 1 --max-jobs 3 \
  --teacher-candidates 256 --teacher-max-steps 65536 --nested-teacher \
  --cluster-attempts 256 --validation-seconds 10800
```

Add `--dry-run` to create and inspect the fresh run directory, DAG and `run_plan.json` without
submitting; this also works with CPU-only Python. `--seeds 1` creates seed-0 continuations;
`--seeds 3` creates seeds 0, 1 and 2 for each method, all sharing one seed-0 warmup.
Add `--decoder discrepancy` to use the alternative construction search described below
during validation and deployment; omitting it preserves the original depth-first order.
Each job requests one GPU and 32 GB host memory; `--max-jobs 3` limits concurrent jobs.
The GPU capability floor is 7.5 for the shared environment; its cu128 PyTorch build cannot run on the P100. A CUDA version string alone does not establish hardware compatibility.

Warmup starts from **seeded random weights, without a checkpoint**, and uses teacher imitation only.
Its validation-selected `warmup/policy.best.pt` starts continued imitation, REINFORCE plus
imitation (weight 1, no further warmup), and Q-learning. Every phase receives ten planned passes
and the same 256-contraction inference allowance. This matches puzzle exposure and allowance,
**not** optimizer updates, oracle measurements or actual contraction work. Q uses batch size 8 and one update before and after each visited state.

With 10,051 pools, a pass is 21 chunks of at most 500 puzzles, ending with 51; ten passes are
210 chunks per phase. Cycling visits every shuffled pool before repeating it. Chunk filenames
retain `epoch` for compatibility. Logs/checkpoints record actual visits, distinct puzzles,
state exposures, updates and passes. Actor teacher caches retain encountered states across passes.

Nested structural teaching preserves proposal prefixes at 32, 64, 128 and 256 candidates,
with traversal budgets 8,192, 16,384, 32,768 and 65,536 steps: 122,880 nominal steps together. The largest-stage limit is not the total. Affordability and duplicate checks can return fewer
proposals. Each uncached state also allows 80 failed-value probes and 32 mutations:

- **Structural search** proposes core sets from network connectivity and cycles.
- **Failed-value probes** temporarily assume a value and trace original cores causing a GAC conflict.
  The resulting cluster is evaluated under the original domains, without that assumption.
- **Mutations** change cores or outputs of promising clusters to explore nearby constructions.

Only contraction/GAC-verified useful constructions supply positive teachers. Requesting 256
proposals does not guarantee 256 useful labels. `structural proposals N on S fresh states (U useful)`
counts structural proposals, fresh states and measured structural successes; probes/mutations are separate.

The helper copies data and submit files into a new run directory. Learner code still runs from
the shared checkout: sync it before submission and keep it stable while those jobs run.
Omitting `--data` deliberately retains the earlier checkpoint-based workflow in
[teacher_branches_gpu.dag](teacher_branches_gpu.dag); schedule/teacher overrides require `--data`.
The main [submit template](teacher_branches_gpu.sub) and older standalone
[Q template](qlearn_gpu.sub) remain available for reproducing their respective workflows.

## Objectives, metrics and checkpoints

The workflow above explicitly selects **pruning**. Its reward is the fraction of remaining
weighted log-domain uncertainty removed by contraction plus GAC, clipped to [0, 1]; a contradiction
earns 1. Skips, rejections and unchanged domains earn 0. Reward colors have weight 1; other
variables receive a bounded auxiliary weight. There is no time/work penalty or extra solve bonus.
Training measurement restores domains; a separately applied useful cluster supplies a new context.

`learn.py` still defaults to **mixed** for older experiments: pruning/solve/contradiction rewards,
work/time penalties and optional log DFS improvement. `--objective dfs_nodes` instead uses
completed ordinary-MAC nodes before minus after the contraction; negative savings remain negative.
Unfinished counterfactual searches are unknown and excluded from that loss and teacher selection.
Q-learning uses the pruning reward and propagates completed-construction rewards through replay.

`sampled useful` measures sampled completed constructions with pruning or a contradiction.
By default, `imitation` is the negative log probability mass of teacher-compatible
continuations. With `--greedy-imitation`, it instead scores an individual compatible
action at corrected greedy prefixes. Lower loss means better fit to those labels;
it does not measure pruning or DFS improvement.
`greedy useful` measures one greedy construction per unresolved validation root.
Repeated root progress and full-solver DFS nodes measure different behaviors; none of these
intermediate metrics alone demonstrates a reduction in test-set DFS nodes.

Full validation runs initially, every four chunks and finally in the scaled workflow.
The latest `.pt` is saved every chunk, immutable `.epochNNNN.pt` snapshots every five,
and `.initial.pt` preserves initialization. Prefer `.best.pt` when its validation
uses the deployment decoder and budgets: with pruning/Q search validation,
completed solves rank first and DFS nodes second; ties keep the earlier weights.
The initialization can remain best. Final weights can perform worse than the saved best.
A limited solve has no completed aggregate DFS total; 10,000 nodes and 10,800 seconds per puzzle
are validation guards in the command above, not reward terms.

GPU training moves neural scoring and gradients to CUDA. Exact contractions, GAC and DFS remain on CPU; validation uses CPU weights to match deployment. Sequential oracle/decoding work can
limit GPU utilization. Submit on bernoulli; CUDA availability is checked on the allocated worker.

## Inference and fair comparisons

The learned policy chooses cores and outputs, rather than selecting a fixed structural candidate. A deterministic decoder follows greedy actions and then policy-ranked alternatives, deduplicating
partial constructions and completed clusters. The default decoder budget is 8,192 scored states
per domain snapshot. Skips end construction branches; construction exhaustion ends the episode.

`--decoder discrepancy` explores alternatives to earlier decisions before exhausting output
variants of one core set. It orders paths by the sum of local action ranks, breaking ties by
the earliest deviation. The first greedy construction is unchanged; core, target, output and
finish decisions still come from the learned scores. Its pending frontier is bounded by twice
the construction-state budget. Existing checkpoints retain `depth_first` unless overridden;
the option is supported by both learners, the evaluation CLI and the HTCondor helper.

`--decoder sampling` keeps the greedy path first, then draws complete paths from
the action probabilities. It uses one encoding per domain snapshot and a local,
domain-seeded generator; global training randomness is unchanged. Every scored
decision counts toward the same 8,192-state budget, including skips, duplicates
and unfinished paths. Candidate duplicates are suppressed. This aligns actor
inference with sampled construction training and is a separate choice from the
learned weights. Production exactly reproduces the validation prototype's 2,048
proposals and all 93,195 scored states. Actor evidence uses CPU evaluation with
beta zero. Q sampling uses `softmax(Q)` at temperature one and remains experimental;
its scale-sensitive probabilities have no solver-performance evidence here.

`--decoder sampling_grow` keeps complete-path sampling but removes finish/skip
while the current phase has a legal core or open-variable extension. It still
learns which cores and outputs to choose; construction size is determined by
the existing masks and bounds. Completed clusters must pass runtime contraction
work and tensor-size checks. This generic alternative tests premature stopping;
it introduces no Sudoku deduction rules and is not a claim that maximal growth
always helps. Free actor training rollouts and greedy `propose` use the same
conditional action distribution as this decoder. Forced teacher traces/prefixes
retain their original scoring API; only a free suffix uses the filter. Existing
decoder modes and old checkpoints keep their original behavior. Q inference can
use this mode experimentally, but Q training/replay masks are unchanged and
there is no Q performance evidence for it.

At a GAC closure, choose a cluster, contract under current domains, apply its temporary summary,
run GAC, and remove the summary. Useful pruning remains on the MAC trail and is restored correctly
on backtracking. After pruning, construct again from the updated domains without resetting the
remaining allowance. Default inference can continue after a no-op; `--stop-on-noop` reproduces
the earlier first-no-op rule. A contraction that does no pruning consumes a slot; rejected
constructions and previously cached no-ops do not. A no-op has no lasting deduction.

The scaled run allows 64 cores, six outputs, 1M estimated work and 100K tensor entries per cluster, with compact domains and no cumulative work cap. CLI defaults can differ; checkpoint config and
explicit overrides determine deployment. The NumPy executor permits at most 52 distinct colors
per cluster. A runtime work cap is per propagation episode; `experiment.py --total-work` caps the
whole solve and falls back to ordinary MAC after exhaustion, retaining earlier pruning.

For a **new** held-out dataset, choose fresh output paths and comparable contraction allowances:

```bash
python -m experiments.cluster_selection.reinforcement.experiment NEW_TEST.json \
  --training-data TRAIN.json --checkpoint NEW_RUN/policy.best.pt \
  --output NEW_EVALUATION/solves.jsonl --indices 0 1 2 3 4 \
  --modes plain heuristic untrained trained \
  --baseline-candidates 32 --cluster-attempts 32 \
  --nodes 10000 --seconds 600 --beta 0 --decoder sampling
```

Add `--propagation-only` with another output path for the no-DFS comparison. Reports include assignment validation, source/input hashes, contractions, estimated work and censored outcomes.
Higher heuristic allowances also require explicitly scaled structural search effort;
**`experiment.py` does not scale search steps automatically**. Merely raising its candidate cap
can change coverage. [ensemble_experiment.py](ensemble_experiment.py) reproduces the archived
frozen-checkpoint comparison and scales heuristic steps as `max(8192, 256 * allowance)`;
it accepts `--allowances` and a new `--output` directory. It follows its archived reference split
and checkpoints, rather than configuring an arbitrary new held-out experiment.
Equal contraction caps do not imply equal search effort, tensor work or wall time.

Evaluation rejects exact overlap with the supplied training corpus and checkpoint exposure
metadata when available; older checkpoints can lack that metadata. Exact disjointness does not
exclude Sudoku-symmetry duplicates. The standard 100-puzzle Sudoku benchmark overlaps the original
training corpus. Our 57-puzzle test split was reused in several comparisons, so it is no longer
untouched test evidence. Keep validation for checkpoint selection and reserve new tests prospectively.

## Data, compatibility and archived experiments

Generic input uses `version: 1`, `network` from `data.network_data`, `initial_domains` (a list of
color-to-domain dictionaries), and optional `reward_colors`, `samples`, `train_indices` and
`validation_indices`. Existing Sudoku `puzzles` use their domain adapter. Without explicit splits,
80% of instances train and 20% validate. Saved contexts must belong to training puzzle indices.
`prepare.py` adds training branch states, not independent puzzles; `scale_data.py` filters fresh
source puzzles using current MAC GAC. Both require new output filenames; see their `--help`.

Existing outputs are rejected: use a fresh experiment path, preserve checkpoints/results and record code versions. `--initial-checkpoint` is a compatible-weight warm start with a fresh optimizer,
**not exact resumption**. Matching Q checkpoints can reuse compatible replay; actor-to-Q warm starts
retain features but reset the final score layer. Preserve architecture/configuration when loading
old checkpoints; do not add value or relational encoding to flat-encoder weights.

Optional diagnostic variants remain available: `--on-policy-imitation` repairs student prefixes;
`--completion-value-imitation` teaches best measured reachable completion utility. They are mutually
exclusive and require target-first construction; repair is not graded for the DFS-only objective.
`--value-encoding` and `--relational-context` test richer representations and symmetry properties.
They are not enabled in the current scaled workflow. Explicit cross-network initialization needs
`--transfer-weights --value-encoding` and compatible parameter shapes; successful transfer is unproven.

`learn.py --greedy-imitation` follows the highest-scoring teacher-compatible continuation and
trains its individual probability. The default instead trains total compatible probability on
a random teacher trace. This new option excludes the repair/completion-value losses. A paired
240-update development trial improved over the matched mass-loss control, but its weights did
not beat the existing teacher with discrepancy decoding. The selected weights therefore remain
unchanged; the frozen selection (`results/constructive_improvement_20261106.903639e9/selection_frozen.json`)
records the comparison. Neither loss labels unmeasured alternatives as proven useless.

Historical evidence is preserved rather than folded into current-run claims:

- Original Sudoku report (`results/useful_clusters_20261021/REPORT.md`) and
  small-problem curriculum (`results/efficiency_curriculum_20261021/REPORT.md`).
- Equal-allowance reruns (`results/equal_cluster_attempts_20261022/REPORT.txt`) and
  DFS-node allowance sweep (`results/node_budget_sweep_20261022/REPORT.txt`).
- Fixed-teacher diagnostic (`results/fixed_teacher_diagnostic_20261028_seed23/REPORT.md`).
- Completed frozen-branch comparison (`results/completed_branches_test_20261102.1d29wvmo/summary.json`)
  and combined learners at allowance 64 (`results/ensemble_allowance64_20261105.b879355b/summary.json`).
- [Learning-strategy explanation](learning_strategies_explained_20261023.tex) and
  [meeting summary](meeting_summary_20261023.tex), retained as dated project notes.
