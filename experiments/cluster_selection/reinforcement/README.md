# Learning useful tensor clusters

The larger-data experiment uses [scaled_sudoku_20261028_seed29/development.json](results/scaled_sudoku_20261028_seed29/development.json): **10,000 new Sudoku puzzles verified unresolved by the current MAC GAC**, on the unchanged tensor network. With the existing puzzles there are 10,092 training indices, including 10,051 unresolved roots, and 23 validation puzzles. Puzzles 85 and 91 are now training puzzles, excluded from validation. The new data supply roots rather than the earlier saved branch samples; successful cluster applications generate further training contexts. Equivalent GAC closures are deduplicated by the loader, so the log's actual pool count determines coverage.

The preparation checked 10,240 source puzzles and retained the first 10,000 unresolved ones in seeded scan order. All came from the 14,321-row `train_1.parquet` shard. The larger `train_0.parquet` shard remains available but was not needed. Source difficulty controls scan priority only; inclusion requires actual current-domain GAC. Source validation and reserved test grids are excluded. Stored solutions are checked against givens and constraints; neither solutions nor teacher usefulness determine policy actions. The earlier AlphaReason filter also used tensor-network GAC, followed by generic forward chaining, on a bounded source prefix; it did not use the current MAC implementation. Its solved/unsolved label is not reused here.

From the shared repository on bernoulli, start a fresh run with the expanded teacher search:

```bash
bash experiments/cluster_selection/reinforcement/submit_teacher_branches.sh \
  /Home/optimier/schuette/.conda/envs/alphareason/bin/python \
  --data experiments/cluster_selection/reinforcement/results/scaled_sudoku_20261028_seed29/development.json \
  --passes 10 --chunk-size 500 --seeds 1 --max-jobs 3 \
  --teacher-candidates 256 --teacher-max-steps 65536 --nested-teacher \
  --cluster-attempts 256 --validation-seconds 10800
```

The helper creates a fresh directory, a concrete DAG and `run_plan.json`. Add `--dry-run` to prepare and inspect these without submitting; dry runs also work with CPU-only Python. With 10,051 pools, one pass is 21 chunks, including a final 51-puzzle chunk. The warmup and each continuation run 210 chunks, covering ten passes. `--cycle-puzzles` visits every shuffled pool before repeating one; it does not independently sample 500 puzzles at each chunk. Logs and checkpoints record actual visits, distinct puzzles, state exposures and optimizer updates. A chunk is retained as an `epoch` in checkpoint/history filenames for compatibility. `--puzzles-per-epoch 0` is also supported for full-pass epochs in direct learner commands.

The nested teacher generates structural pools at 32, 64, 128 and 256 proposals, appending distinct clusters while preserving earlier prefixes. The corresponding traversal budgets are 8,192, 16,384, 32,768 and 65,536 steps. `--teacher-max-steps 65536` sets the **largest stage's budget**; the four stages nominally allow 122,880 steps together. `run_plan.json` records these stages and their nominal sum. These are search limits, so fewer clusters can be returned after affordability and duplicate checks. The existing 80 failed-value probes and 32 mutation attempts per uncached state remain unchanged. Every teacher is measured by contraction under current domains followed by GAC; only measured usefulness supplies positive training examples. The experiment tests whether broader teacher coverage improves learned construction, which can still produce clusters outside those proposals.

The chunk log `structural proposals N on S fresh states (U useful)` reports actual structural proposals across uncached teacher states and how many passed legality checks and measured as useful. Cache hits do not count as new teacher generation; failed-value probes and mutations are outside these structural counts.

Warmup starts from **seeded random initialization (seed 0), with no checkpoint loaded**, and trains with teacher imitation only for ten passes. Its validation-selected `warmup/policy.best.pt` then starts three separate ten-pass branches: continued imitation, REINFORCE plus imitation, and Q-learning, all with seed 0 in the command above. Increase `--seeds` to repeat each branch with more seeds. REINFORCE has zero additional warmup and retains imitation weight one. Every phase receives the same planned puzzle exposure and 256-cluster inference allowance. Q uses batch size eight and one update before and after each visited state; actor updates, oracle calls and explored trajectories differ, and are recorded. This is not an equal-gradient-count comparison. Actor teacher caches retain measured states across passes (`--teacher-cache-size 0`); each job requests 32 GB host memory. At most three single-GPU jobs run concurrently after the shared warmup. The existing GPU compatibility requirement is preserved.

Omitting the new search options preserves the earlier defaults: one structural generation pass capped at 64 proposals and 8,192 steps, 32 inference attempts, and a 600-second validation guard. `--teacher-max-steps auto` (the default) resolves to 8,192 without nesting, or 256 times the smallest power of two covering both 32 and the requested teacher count with nesting. Explicit schedule and teacher/inference overrides require `--data`; invoking the helper without it keeps the legacy workflow below.

Sync the updated repository before submitting, including both learners, candidate generation, contraction planning, the helper and submit template. Generated run directories keep their own DAG and submit-file copies, so later edits to the helper or template do not alter their saved arguments. Learner code still runs from the shared repository; keep that checkout stable while its DAG jobs run. The original scaled dry-run and integration records below used the earlier checkpoint-based warmup and remain preserved as historical records.

The corrected workflow's [argument verification](results/teacher_forks_scaled_20261028.9y0imvwd/verification.json) checks all ten scaled jobs and the legacy jobs through their actual learner CLIs. The scaled warmup has no `--initial-checkpoint`; every child loads the new warmup's selected checkpoint. A separate [fresh-warmup check](results/scaled_sudoku_20261028_seed29/fresh_warmup_smoke/summary.json) verifies the initial weights equal a seed-0 random policy and that teacher updates change them.

The latest checkpoint is saved every chunk, immutable snapshots every five chunks, and full validation initially, every four chunks and finally. The command above validates greedy usefulness, repeated propagation and completed DFS nodes with 256 cluster attempts, a 10,000-node guard and a 10,800-second guard per puzzle. The longer wall guard avoids interpreting slow inference as a DFS-node result; searches that hit either guard have no aggregate node total. The 23 retained puzzles are development validation. Previously reported test puzzles must not become checkpoint-selection data. GPU optimization and CPU contractions/GAC remain separate. This local machine has no GPU or HTCondor client, so the workflow is prepared and tested locally but must be submitted on bernoulli.

The [preparation and verification record](results/scaled_sudoku_20261028_seed29/protocol.json) includes hashes, counts and references. A separate 32-root [teacher pilot](results/scaled_sudoku_20261028_seed29/teacher_pilot.json) found 100 useful, affordable clusters on 19 roots, without teacher-based data filtering. Local verification passed 143 tests, with two CUDA-only cases skipped; every generated job parsed with its actual learner CLI. A bounded [integration run](results/scaled_sudoku_20261028_seed29/integration_smoke/summary.json) exercised new Sudoku roots, cycling, warmup and both reinforcement continuations. Every method solved its two validation puzzles in four total DFS nodes; this checks execution, not improved learning. For the retained 23 validation puzzles, earlier completed results give plain MAC 315 nodes, the structural heuristic 188, and the warm-start actor 292. Those are comparison references, not results of training on the larger dataset.

To regenerate the subset into a fresh path:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m experiments.cluster_selection.reinforcement.scale_data \
  experiments/cluster_selection/reinforcement/results/teacher_branches_20261023/development.json \
  --dataset-dir demonstrations/sudoku/sudoku_bench/sample_data/hf_sudoku_dataset \
  --output NEW_SCALED_DATA.json --target 10000 --workers 8 --batch-size 64 --seed 29 \
  --reserved-data experiments/cluster_selection/data/testing.json
```

The following describes the earlier, smaller experiment and its original search defaults. Invoking the submission helper without `--data` preserves that workflow.

The shared-start experiment uses the successful target-first actor checkpoint as its parent: train imitation first, freeze the validation-selected checkpoint, then compare continued imitation, REINFORCE plus imitation, and Q-learning. The seven-core construction solving validation puzzle 96 appeared during imitation warm-up; it does not isolate a reinforcement-learning benefit. The new experiment keeps the same flat contextual architecture and tests the learning rule rather than adding another encoder.

The prepared input is `results/teacher_branches_20261023/development.json`. It retains the existing 90 training and 25 validation puzzles and adds generic GAC-closed MAC branch contexts from training puzzles only. These are more training states, not new independent Sudoku puzzles. The previously untouched test puzzles remain outside this preparation and all training/selection. To regenerate into a **fresh** path:

```bash
python -m experiments.cluster_selection.reinforcement.prepare \
  experiments/cluster_selection/reinforcement/results/useful_clusters_20261021/expanded_development.json \
  --output NEW_DATA.json --states-per-puzzle 16 --nodes 512 --seed 0
```

The saved preparation expands the effective pool from **114 to 747 unique training contexts** across 49 unresolved training puzzles. It adds 633 new branch contexts and serializes 14 root closures the learner already supplied automatically; the original 100 saved states are preserved. All 747 contexts were independently checked for GAC closure, uniqueness and consistency with their own training puzzle's givens. The other 41 training roots already solve with ordinary GAC.

The [shared-filesystem workflow](teacher_branches_gpu.dag) uses [one submit template](teacher_branches_gpu.sub). It first runs 30 teacher-only epochs from `policy_goal_actor_seed0.best.pt`, then 60 additional epochs for each continuation with seeds 0, 1 and 2: ten jobs in total. All nine children load `warmup/policy.best.pt` after the parent finishes. If warm-up never improves validation DFS results, this selected file can retain the parent's original weights. Branches load weights into fresh optimizers; Q-learning preserves the encoder/context and resets its final score layer because logits are not calibrated Q-values. This is a comparison from a common learned representation, not identical initial actor/Q outputs.

All arms use actual normalized pruning rewards (`--objective pruning` for actors), the same input split, per-cluster limits and 32 inference attempts. REINFORCE retains imitation weight one and uses unforced constructions (`--warmup 0 --no-curriculum`). Q-learning uses replay and epsilon-greedy exploration. Their optimizer update counts, trajectories and teacher measurements differ; equal epochs are not a matched number of oracle calls or gradient updates. Record those counts rather than claiming this is a pure loss-only ablation. Training uses up to 90 puzzle pools per epoch, four rollouts and three successive states, with 64 structural teachers, 80 failed-value probes and 32 mutations per uncached state. Affordable clusters can use up to 64 cores. Contraction work/size limits remain 1M/100K per cluster; cumulative work caps are disabled. No tensor inference or Sudoku rule is replaced by a neural guess.

Full validation runs initially, every ten epochs and at the final epoch, using 10,000-node/600-second guards per solve. CPU validation includes single greedy-construction usefulness, repeated cluster propagation, and full DFS; incomplete searches have no aggregate node total. `.best.pt` ranks completed solves then DFS nodes, keeping earlier checkpoints on ties. `.pt` saves every epoch, and `.epochNNNN.pt` saves every five; skipped validations are explicitly marked rather than carrying stale results forward. Both learners support `--device cuda`, `--validation-interval` and `--checkpoint-every`. The GPU trains the network; exact contractions, GAC and solver validation remain on CPU.

Under this pruning objective, training does not run counterfactual DFS for every sampled cluster. The actor log therefore labels node-saving trials **ungraded**, while periodic validation still reports completed solver nodes. Sampled usefulness, teacher-forced success and greedy root usefulness are different metrics; neither loss nor pruning success alone demonstrates improved DFS performance.

Run this on the cluster submission node after activating a **shared CUDA-enabled Python environment**, with this repository installed using `python -m pip install -e .`. The repository must include the prepared JSON and old warm-start checkpoint. Add any site-specific accounting requirements to the submit template first. The submission helper resolves the active interpreter to its real absolute path, imports the learners and rejects CPU-only PyTorch or missing files before submitting. It does not require a GPU on the login node:

```bash
# From the shared tnreason-py directory, after activating your environment:
bash experiments/cluster_selection/reinforcement/submit_teacher_branches.sh
```

An explicit Python executable may be passed as the helper's first argument. It creates a fresh dated run directory and prints that directory and its monitoring command. There are no example executable paths to accidentally submit. `-maxjobs 1` in the helper runs one single-GPU job at a time; increase it to run several branches concurrently. The [DAGMan options](https://htcondor.readthedocs.io/en/main/man-pages/condor_submit_dag.html) pass the shared paths to the workflow and limit submitted nodes. Each branch has its own logs and checkpoint directory. After eviction or interruption, warm-start a new run from a saved checkpoint; exact optimizer resumption is not implemented. GPU execution and scheduler submission cannot be tested on this local CPU-only machine, which has no HTCondor client.

The template requests compute capability **7.5 or newer**, which avoids the P100 assigned to the failed first submission. The existing workspace log `gpu_check/gpu_check.err` already records that `alphareason`'s PyTorch `2.11.0+cu128` supports `sm_75` and newer, and fails on the P100 (`sm_60`) with “no kernel image is available.” A CUDA version string alone does not establish GPU compatibility. If only P100 GPUs are available, use a PyTorch CUDA 12.6 build supporting `sm_60` and adjust `require_gpus` to permit capability 6.0. See the [PyTorch 2.8 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.8.0) and [CUDA support matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md). The new GPU requirement can leave jobs idle if this pool has no matching newer cards.

For the **26 October failed submission**, the logs identify DAGMan `7105.0` and WARMUP `7106.0`. Python never started because its executable was the literal placeholder `/shared/path/to/cuda-environment/bin/python`; no checkpoints were created. Exporting a corrected value in a new shell does not update the running DAG's captured environment. On `bernoulli`, remove that failed DAG and start a fresh workflow with the confirmed real interpreter:

```bash
condor_rm 7105.0
cd /Home/optimier/schuette/tnreasonproj/tnreason-py
bash experiments/cluster_selection/reinforcement/submit_teacher_branches.sh \
  /Home/optimier/schuette/.conda/envs/alphareason/bin/python
```

These commands preserve the old run directory and logs. Removing the DAG also removes its held WARMUP child. Merely releasing `7106.0` keeps its wrong executable; repairing that one job alone would leave the later nodes with the old captured environment. This recovery applies to the logged pre-execution failure, not to resuming a partially trained run.

Local verification: 444 cluster/MAC tests passed, with nine CUDA cases skipped. All ten workflow argument sets parse with the actual learner CLIs. A two-epoch generic [shared-parent smoke](results/teacher_branches_20261023/smoke/summary.json) confirms both actor initializations preserve the parent, Q initialization preserves its features while resetting the final score head, and the parent stays unchanged. Plain MAC uses three DFS nodes; parent, heuristic and all three continuations use one. They are already at the node floor, so this validates integration rather than a learning gain. The planned settings and hashes are in [protocol.json](results/teacher_branches_20261023/protocol.json).

After all arms are frozen, compare their saved validation DFS totals and then run a single predeclared held-out evaluation with plain MAC, the common teacher parent and the generic structural heuristic. The ordinary 32-proposal heuristic is appropriate for this workflow's 32-attempt validation. Larger final allowances need the staged structural pool described below; blindly requesting more structural proposals can change the generator's coverage. The data and warm-up use already examined development puzzles, so validation improvements remain development evidence.

`qlearn.py` trains a constructive action-value policy using replay and terminal pruning, following the graph-construction formulation in [Dai et al. (2017)](https://papers.nips.cc/paper_files/paper/2017/file/d9896106ca98d3d05b8cbdf4fd8b13a1-Paper.pdf). `learn.py` retains the earlier teacher/imitation plus REINFORCE learner, and `train.py` reproduces the oldest experiments. All reuse `ConstraintNetwork`, MAC states and trails, tensor encoding, and the domain-conditioned contraction evaluator. The supervised learner is unchanged.

In Q-learning, one episode starts with empty core/output selections at fixed current domains. The policy chooses a target, connected cores, additional outputs and finish/skip. Intermediate rewards are zero; completion receives the fraction of weighted log-domain entropy removed by contraction and GAC, with contradiction worth one and a completed no-op worth zero. There are no solve, work or timing bonuses. Gamma is one, so longer affordable constructions have no discount penalty. Q values are raw head outputs; actor-specific stop biases, structural guidance and probability normalization do not define their values.

Replay retains complete legal traces and measured outcomes for successes, no-ops and rejected constructions, including resulting domains. The 3,200 saved candidates in `expanded_development.json` are re-evaluated as proposals under current domains and limits, rather than trusting old gains. Replay samples puzzles uniformly, oversamples successful episodes and terminal decisions, and mixes these with fresh epsilon-greedy constructions. Fresh episodes always start from an empty cluster: there are no forced teacher prefixes. The policy can choose any legal core or output, beyond the saved proposals.

The value update uses a frozen target network and undiscounted masked backups. Only the actual terminal action has an exact terminal target. Unfinished prefixes bootstrap legal continuations; measured successful returns are achievable lower bounds. Multi-step successors on successful traces can strengthen the target, while a weaker completion cannot erase a better one-step continuation. This adapts fitted Q-learning to a sparse contraction oracle; it is not an exact reproduction of Dai et al.'s objectives. A failed completion therefore does not label all earlier construction actions useless.

Run the new learner from the repository root, with a fresh output path:

```bash
SOURCE=experiments/cluster_selection/reinforcement/results/useful_clusters_20261021
RUN=experiments/cluster_selection/reinforcement/results/q_learning_20261022_seed0
python -u -m experiments.cluster_selection.reinforcement.qlearn \
  "$SOURCE/expanded_development.json" --output "$RUN/policy.pt" \
  --initial-checkpoint "$SOURCE/hard_validation/selected_trained.frozen.pt" \
  --width 32 --rounds 2 --seed 0 --epochs 8 --puzzles-per-epoch 49 \
  --rollouts 4 --steps 3 --teacher-candidates 32 --teacher-mutations 16 --teacher-probes 40 \
  --batch-size 32 --updates-per-state 16 --n-step 8 --target-update 100 \
  --replay-capacity 8192 --max-cores 64 --cluster-attempts 512 \
  --search-nodes 10000 --validation-seconds 600
```

There is no cumulative work cap in this command; per-cluster work/size caps remain. For API callers, `max_total_work` caps validation/deployment; each training episode independently evaluates one cluster. The optional actor warm start preserves tensor/context features and resets the final scoring layer for Q values. Omit it for fresh initialization; do not add value/relational encoding when loading this particular checkpoint. A matching Q warm start can reuse saved replay when dataset, reward semantics, tensor network and complete model configuration match. It starts a fresh optimizer; this is not exact training resumption.

Q checkpoints include replay, target weights and optimizer data alongside the normal inference weights. `.best.pt` is selected by completed validation solves and then DFS nodes, with no pruning/time tie-breaker. It may remain the initialization. Logs additionally show fully unforced training usefulness, a fixed validation first-construction usefulness probe, and full-solver DFS nodes. Per-instance statuses and node counts identify censored validation solves. These development diagnostics do not establish held-out Sudoku improvement.

For GPU training, add `--device cuda`; CPU remains the default. The Q model, frozen target and gradient updates move to the assigned CUDA device. Tensor contractions, GAC and DFS remain on CPU. Validation uses a CPU copy of the current weights, matching normal checkpoint deployment. Requesting CUDA without an available CUDA-enabled PyTorch installation raises an error before creating run files. GPU checkpoints can still be loaded on CPU. Small sequential construction operations can limit GPU utilization; GPU allocation does not move the contraction oracle onto the GPU.

The shared-filesystem HTCondor template is [qlearn_gpu.sub](qlearn_gpu.sub). It requests one GPU, one CPU and 16 GB host memory, and uses job IDs for unique checkpoint directories and logs. Both the repository (including the development JSON and frozen warm start) and the Python environment must be accessible at the same paths on execute nodes. Install a CUDA-enabled PyTorch build appropriate for the cluster using the [official selector](https://pytorch.org/get-started/locally/), then install this repository into that environment:

```bash
# First activate the shared CUDA environment/load any required cluster modules.
# Run on the cluster's submission/login node, in the shared repository.
GPU_PYTHON="$(python -c 'import sys; print(sys.executable)')"
"$GPU_PYTHON" -m pip install -e .
# On a CPU-only login node, a CUDA-enabled build still reports a nonempty
# torch.version.cuda; is_available() is only expected true inside a GPU job.
"$GPU_PYTHON" -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.version.cuda is not None, "CPU-only PyTorch"' &&
condor_submit repo_root="$PWD" python_exec="$GPU_PYTHON" \
  experiments/cluster_selection/reinforcement/qlearn_gpu.sub
```

The Python environment used locally during development has CPU-only PyTorch 2.4.1; requesting a GPU does not change that build. Torch is not installed by this repository's setup dependencies. An existing cluster CUDA environment is suitable; a new shared virtual environment can be created before following PyTorch's installation instructions. The job preserves HTCondor's assigned `CUDA_VISIBLE_DEVICES`, and the training log prints its GPU and CUDA runtime. See [HTCondor's GPU documentation](https://htcondor.readthedocs.io/en/main/users-manual/submitting-a-job.html#jobs-that-require-gpus).

Results are written below `experiments/cluster_selection/reinforcement/results/q_gpu_CLUSTER_PROCESS/`; `.out`, `.err` and `.log` files sit next to that directory. Use `condor_q` to inspect the job and `tail -f` on its `.out` file to follow progress. Failed jobs are held so their error can be inspected. The template requires no local wrapper and transfers no files; add any account or GPU restrictions required by the cluster. Pass `run_seed=1` to choose another seed. Existing output protection still applies after eviction/restart: to continue, submit a new job with `checkpoint_path=/shared/path/to/saved/policy.pt`. Its job ID supplies a fresh output directory. This is a fresh-optimizer warm start with compatible replay, not exact optimizer resumption. This workspace has neither CUDA hardware nor `condor_submit`, so hardware execution and scheduler submission have not been tested here.

Both learners encode selected cores/outputs and their incidence structure. `learn.py --target-first` chooses an active output variable, grows connected cores incident to it, then chooses further outputs and finishes; Q-learning uses this ordering by default. Core choices are not restricted to structural proposals. Structural proposals and measured augmentations supply training teachers; saved new policies use `beta=0` and construct directly at inference.

Use `learn.py --objective dfs_nodes` to optimize search: a completed trial earns **ordinary MAC DFS nodes before minus nodes after** its contraction and GAC. Negative savings remain negative rewards. Skips, rejected contractions and no-ops earn exactly zero because their domains are unchanged. There are no pruning, solve, contradiction, work or timing bonuses in this reward. Both searches must finish: a node/time limit makes the reward unknown, excluded from the actor loss, its baseline and scale estimate, and teacher selection. Only verified positive savings supply teachers and training-state advances; pruning-only teachers are not a fallback. Repeated states share cached searches. The small entropy term remains a policy regularizer.

The default `learn.py --objective mixed` preserves earlier experiments: normalized log-domain reduction, solve/contradiction bonuses, work/time deductions, and optionally `log((nodes_before+1)/(nodes_after+1))` when `--search-nodes` is positive. Its useful-pruning teacher fallback is preserved. Changing only `--search-nodes` therefore does not enable the DFS-only objective.

Training starts from puzzle-balanced GAC closures and saved training states. Useful constructions supply further states. In `learn.py`, measured teachers provide a continuation-mass imitation loss, followed by immediate REINFORCE on sampled constructions; a shrinking forced prefix makes the first successful trials reachable. Initial target labels include only teacher outputs that actually shrink after GAC, while the complete output set is retained. This is single-state credit with a state curriculum, not a full solver-episode policy gradient.

Two optional `learn.py` losses address errors on the policy's own construction prefixes. `--completion-value-imitation` also credits actions leading to the highest measured terminal utility still reachable from each greedy prefix, including decisions to finish. Exhaustive teachers give exact completion labels on tiny problems; sampled teachers give lower bounds and can miss other useful completions. `--on-policy-imitation` instead repairs student extensions by adding a measured teacher's cores and retaining all its outputs, checking affordability without extra contractions. These modes require `--target-first` and are mutually exclusive. Repair imitation is rejected for `dfs_nodes`: the repaired union has not itself been graded for search savings. The repair variant did not improve the reserved 4×4 result over ordinary training.

`--teacher-probes N` adds up to N failed-value GAC probes per uncached training state. A scratch copy traces source cores responsible for a conflict; those cores form a proposed cluster retaining the assumed variable. Every proposal must still pass construction masks and actual domain-conditioned contraction/GAC measurement before receiving a positive label. These are offline training probes, never inference actions. The probe count, candidate count and extraction seconds are recorded separately. Larger `--max-cores` can retain affordable explanations: a training-only 9×9 audit verified useful 33–64-core clusters costing less than 2,000 estimated work. The current NumPy backend still restricts each cluster to 52 distinct variables.

`--value-encoding` replaces the RL policy's flat tensor encoder with messages between active variable values and domain-compatible allowed tuples. It shares parameters across value labels and dimensions, so independent value renamings preserve scores up to floating-point rounding. It leaves the shared encoder and supervised learner unchanged. This is an experimental representation, not a complete contraction algorithm: some regular permutation cycles still have identical embeddings despite different pruning outcomes.

`--relational-context` adds selected-cluster value/tuple messages and one query per active value of the chosen target. A scalar anchor signal diffuses through selected allowed tuples, and its first and second moments enter possible extensions before values and queries are averaged. This preserves a correlation that ordinary mean pooling loses. The regular-cycle regression now distinguishes extensions pruning zero versus six values, while value/core/axis symmetry tests still pass. This requires `--value-encoding --target-first`; `--relational-rounds` defaults to six. It is a separate opt-in representation so earlier policies keep their original behavior. Passing this capacity test alone does not establish better learning or Sudoku transfer.

For comparison, retrain the earlier DFS-reward learner using a **new** output directory:

```bash
SOURCE=experiments/cluster_selection/reinforcement/results/useful_clusters_20261021
RUN=experiments/cluster_selection/reinforcement/results/dfs_nodes_20261022_seed0
python -u -m experiments.cluster_selection.reinforcement.learn \
  "$SOURCE/expanded_development.json" \
  --output "$RUN/policy.pt" \
  --initial-checkpoint "$SOURCE/hard_validation/selected_trained.frozen.pt" \
  --objective dfs_nodes --target-first --completion-value-imitation \
  --width 32 --rounds 2 --seed 0 --epochs 8 --warmup 1 \
  --puzzles-per-epoch 49 --rollouts 3 --steps 3 \
  --teacher-candidates 32 --teacher-mutations 16 --teacher-probes 40 \
  --max-cores 64 --max-open-colors 6 --max-work 1000000 --max-size 100000 \
  --cluster-attempts 512 --no-total-work-limit \
  --search-nodes 10000 --search-seconds 120 --validation-seconds 300
```

This warm start has width 32, two rounds and target-first contextual encoding; do not add `--value-encoding` or `--relational-context` when loading these weights. Omit `--initial-checkpoint` for a fresh initialization with otherwise identical settings. The data explicitly split 90 training and 25 validation puzzles; 49 training roots remain unresolved after GAC. They are development data, not a fresh held-out evaluation. The 64-core limit admits larger affordable explanations already found in training; keep 32 for an objective-only ablation. `--steps 3` controls successive training states, while `--cluster-attempts 512` controls inference contractions at each GAC closure. The latter is stored in the checkpoint and used by `runtime.attach` unless explicitly overridden. `--no-total-work-limit` disables cumulative work caps in training advances and validation; per-cluster work and tensor-size caps still apply. Search seconds and nodes are guards, not reward terms. If `search_limited` is frequent or validation `search_nodes` is null, raise the guards before interpreting learning curves.

The input contains `version: 1`, `network` from `data.network_data`, `initial_domains` as a list of color-to-domain dictionaries, and optional `reward_colors`, `samples`, `train_indices`, and `validation_indices`. Existing Sudoku `puzzles` inputs use their existing domain adapter. No Sudoku rules enter the policy or reward. Without explicit splits, the first 80% of instances train and the rest validate. Splits and warm starts reject exact instance overlap using canonical domain fingerprints. `--initial-checkpoint` loads compatible contextual weights into a fresh run and optimizer.

For explicit `learn.py` transfer to a different network, add `--transfer-weights --value-encoding` together with `--initial-checkpoint`. The source must also use value encoding; all learned parameter shapes must match, including the maximum tensor arity used by construction context. This supports binary-core 4×4-to-9×9 initialization, but does not establish successful Sudoku transfer. Exposure metadata is retained and the optimizer starts fresh.

Each run preserves `.initial.pt`, `.best.pt`, and the latest `.pt`; an existing run path is rejected. Deploy `policy.best.pt`. In `dfs_nodes` mode validation is required, completed validation solves rank first, and total DFS nodes rank second. Equal-node results retain the earlier checkpoint regardless of pruning or time; incomplete searches cannot appear cheap through missing counts. The initial checkpoint can remain best. In mixed mode propagation progress/time still break search ties, or select the checkpoint without search grading. History includes valid/censored trials, positive teachers, node-saving/worsening trials, net nodes saved, and completed validation nodes. Record data splits, seeds and code versions alongside a run, as in the experiment manifests below.

Use a saved policy without DFS:

```python
from experiments.cluster_selection.runtime import attach

state = network.create_mac_state(initial_domains)
attach(state, "NEW_POLICY.best.pt")  # Uses checkpoint cluster_attempts (legacy default: 32).
state.propagate()
```

At each GAC closure, the policy can apply up to its configured allowance of affordable cluster contractions, including after a no-op. A deterministic decoder follows greedy actions first, then explores policy-ranked alternatives in the same construction action space. It deduplicates partial states and completed core/output sets; it does not select from structural candidates. A skip ends one construction branch. Rejected constructions and cached no-ops do not consume contraction slots. Enumeration is bounded by 8,192 scored construction states per domain snapshot. After useful pruning, it restarts from updated domains without resetting the remaining contraction allowance. Solution, contradiction, exhausted construction search, or contraction/work limits stop the closure. Summary cores are removed; domain pruning stays on the MAC trail and is restored on backtracking. Compact contractions slice tensors to current domains and expand only the resulting summary axes for GAC. Defaults allow 32 cores, six outputs, 1M estimated work and 100K tensor entries per cluster; these are configurable. Cyclic construction masks exclude conditioned incidence forests, which cannot improve already closed GAC.

For propagation and DFS comparisons on a new held-out dataset:

```bash
python -m experiments.cluster_selection.reinforcement.experiment TEST.json \
  --training-data TRAIN.json --checkpoint NEW_POLICY.best.pt --output NEW_RESULTS.jsonl \
  --indices 0 1 2 3 4 5 6 7 8 9 \
  --modes plain untrained trained \
  --baseline-candidates 512 --cluster-attempts 512 \
  --nodes 10000 --seconds 300 --beta 0
```

This uses 512 attempts and no cumulative work limit for both learned and random weights, with plain MAC as another control. Including `heuristic` in `--modes` uses the ordinary generator: requesting 512 candidates at fixed generation effort can starve Sudoku cycle discovery. A meaningful structural comparison needs the staged generation protocol below, preserving the original 32-candidate prefix before expanding the pool. The saved allowance alone does not configure that generator.

Add `--propagation-only` and use another output file for the no-DFS comparison. The evaluator validates full assignments against the original cores and givens, rejects exact overlap with both the supplied training corpus and recorded checkpoint exposure, and records source hashes, work, contractions, timeouts and node limits. Legacy checkpoints without exposure metadata retain only the supplied-corpus check, which the manifest identifies. Its work cap applies across the whole solve; exhausting it retains pruning and continues ordinary MAC. The runtime policy's configured work cap applies to each closure. Construction can apply up to `--baseline-candidates` affordable clusters (32 by default), matching the structural batch's maximum, independently of training teacher counts; `--cluster-attempts` can override the learned allowance. `--stop-on-noop` reproduces the earlier greedy inference loop for historical comparisons. Single-proposal custom adapters, including First, retain their original stop rule. Equal caps do not imply equal work consumed; compare recorded work and wall time as well as nodes. Earlier reports used the first-no-op stopping rule and retain their original results.

Experiments and their limitations are recorded in [the Sudoku experiment report](results/useful_clusters_20261021/REPORT.md) and [the small-problem curriculum report](results/efficiency_curriculum_20261021/REPORT.md). Standard uniquely solvable 4×4 clue puzzles already solve with the existing GAC representation; the small Sudoku curriculum therefore uses partial-domain states with measurable unsupported values. These states have multiple solutions and test useful pruning and DFS reduction, rather than propagation to a unique assignment. The default 9×9 Sudoku benchmark's 100 puzzles all occur in the original training corpus, so its saved results are exploratory evidence rather than held-out generalization.

[Equal-allowance reruns](results/equal_cluster_attempts_20261022/REPORT.txt) use the frozen checkpoints and previously examined reserved inputs with up to 32 affordable contractions for both learned and structural policies. The random-graph trained model solves 25/48 roots without DFS versus 9/48 for initialization and 48/48 for the heuristic. On partial-domain Sudoku, extra attempts leave the trained checkpoint at 119 DFS nodes versus 123 for initialization and 115 for the heuristic, with increased inference cost. No weights were updated or checkpoints reselected.

The [DFS-node allowance sweep](results/node_budget_sweep_20261022/REPORT.txt) removes cumulative-work limits and compares 32, 128 and 512 contractions with frozen weights. Trained generic-graph nodes fall 71→50→48, matching the heuristic at 512; trained Sudoku4 nodes remain 119 while the heuristic changes 115→115→114. More core-set diversity on development states increases pruning without saving nodes. These are diagnostics on previously examined inputs, not fresh held-out results. The corrected Sudoku runner preserves the original 32-proposal prefix and extends its generation effort; simply requesting a larger pool with fixed generation effort can starve cycle discovery. See `results/node_budget_sweep_20261022/compare.py` and `compare_sudoku.py` for the separate reproducible protocols. No production defaults or checkpoint weights were changed by this sweep.

The earlier DFS-only implementation passed 384 cluster/MAC tests, including negative savings, censored-label exclusion, checkpoint ties and backtracking restoration. Its three-epoch [command-line smoke run](results/dfs_objective_smoke_20261022/summary.json) saves positive node-savings labels and reloads the 512-attempt/no-cumulative-cap policy. On its tiny validation CSP, plain MAC uses three nodes and initial/trained/heuristic policies each use one: this verifies that pipeline, not a learning improvement or Sudoku generalization. The subsequent 9×9 DFS-reward run in `results/dfs_nodes_20261022_seed0` completed eight epochs; the selected checkpoint remained the initialization, so it did not establish improved solving.

The earlier three-epoch [Q-learning smoke run](results/q_learning_smoke_20261022/summary.json) uses a small generic tensor CSP with distinct training/validation domains. Greedy validation construction changes from no pruning to useful pruning, and fully unforced training trials change from 0/2 to 2/2 useful. With one contraction allowed for both methods, initialization uses two DFS nodes and the latest Q policy uses one, tying the structural heuristic. At allowances 32 and 512, initialization, trained policy and heuristic all reach the one-node floor, so DFS-based selection correctly keeps initialization. Replay preserves both zero and positive outcomes. This checks learning and solver integration; it does not establish Sudoku generalization. A subsequent full 9×9 run in `results/q_learning_20261022_seed0` completed eight epochs: greedy usefulness remained 0–1/17, and time-limited validation left aggregate DFS totals unavailable. See the preserved `qtraining.out`; these results do not demonstrate a Sudoku learning gain.
