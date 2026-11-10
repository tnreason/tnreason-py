#!/usr/bin/env bash
# Shared-filesystem GPU training; --dry-run only renders a fresh submission.
set -euo pipefail
cluster_python_command=python
cluster_python_set=0
cluster_data=""
cluster_passes=10
cluster_chunk_size=500
cluster_seeds=3
cluster_max_jobs=""
cluster_teacher_candidates=64
cluster_teacher_max_steps=auto
cluster_nested_teacher=0
cluster_cluster_attempts=32
cluster_validation_seconds=600
cluster_dry_run=0
cluster_schedule_options=0
cluster_invocation_dir=$PWD
while (( $# )); do
    case "$1" in
        --data|--passes|--chunk-size|--seeds|--max-jobs|--teacher-candidates|--teacher-max-steps|--cluster-attempts|--validation-seconds)
            if (( $# < 2 )); then echo "Missing value for $1" >&2; exit 2; fi
            case "$1" in
                --data) cluster_data=$2 ;;
                --passes) cluster_passes=$2; cluster_schedule_options=1 ;;
                --chunk-size) cluster_chunk_size=$2; cluster_schedule_options=1 ;;
                --seeds) cluster_seeds=$2; cluster_schedule_options=1 ;;
                --max-jobs) cluster_max_jobs=$2 ;;
                --teacher-candidates) cluster_teacher_candidates=$2; cluster_schedule_options=1 ;;
                --teacher-max-steps) cluster_teacher_max_steps=$2; cluster_schedule_options=1 ;;
                --cluster-attempts) cluster_cluster_attempts=$2; cluster_schedule_options=1 ;;
                --validation-seconds) cluster_validation_seconds=$2; cluster_schedule_options=1 ;;
            esac
            shift 2 ;;
        --dry-run) cluster_dry_run=1; shift ;;
        --nested-teacher) cluster_nested_teacher=1; cluster_schedule_options=1; shift ;;
        -h|--help)
            echo "Usage: bash submit_teacher_branches.sh [PYTHON] [--data PATH --passes 10 --chunk-size 500 --seeds 3] [--max-jobs N] [--dry-run]"
            echo "Scaled teacher/inference options: --teacher-candidates 64 --teacher-max-steps auto --nested-teacher --cluster-attempts 32 --validation-seconds 600"
            exit 0 ;;
        --*) echo "Unknown option: $1" >&2; exit 2 ;;
        *)
            if (( cluster_python_set )); then echo "Only one Python executable may be supplied." >&2; exit 2; fi
            cluster_python_command=$1; cluster_python_set=1; shift ;;
    esac
done
if [[ -z "$cluster_data" && "$cluster_schedule_options" == 1 ]]; then
    echo "Schedule and teacher/inference overrides require --data." >&2; exit 2
fi
if [[ -z "$cluster_max_jobs" ]]; then
    if [[ -n "$cluster_data" ]]; then cluster_max_jobs=3; else cluster_max_jobs=1; fi
fi
for cluster_count in "$cluster_passes" "$cluster_chunk_size" "$cluster_seeds" "$cluster_max_jobs" \
    "$cluster_teacher_candidates" "$cluster_cluster_attempts" "$cluster_validation_seconds"; do
    if [[ ! "$cluster_count" =~ ^[1-9][0-9]*$ ]]; then
        echo "Counts and validation seconds must be positive integers." >&2; exit 2
    fi
done
if [[ "$cluster_teacher_max_steps" != auto && ! "$cluster_teacher_max_steps" =~ ^[1-9][0-9]*$ ]]; then
    echo "--teacher-max-steps must be auto or a positive integer." >&2; exit 2
fi
cluster_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CLUSTER_REPO_ROOT=$(cd -- "$cluster_script_dir/../../.." && pwd)
if [[ -n "$cluster_data" && "$cluster_data" != /* ]]; then cluster_data="$cluster_invocation_dir/$cluster_data"; fi
cd -- "$CLUSTER_REPO_ROOT"
export CLUSTER_GPU_PYTHON
CLUSTER_GPU_PYTHON=$("$cluster_python_command" -c 'import sys; print(sys.executable)')
if [[ ! -x "$CLUSTER_GPU_PYTHON" || "$CLUSTER_GPU_PYTHON" != /* ]]; then
    echo "Python must be an executable at an absolute shared-filesystem path." >&2; exit 1
fi
"$CLUSTER_GPU_PYTHON" - "$cluster_dry_run" <<'PY'
import sys
import torch
from experiments.cluster_selection.reinforcement import learn, qlearn
if sys.argv[1] != "1" and torch.version.cuda is None:
    raise SystemExit("CPU-only PyTorch: activate a shared CUDA-enabled environment before submitting.")
print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
print("CUDA device availability is checked on the allocated worker, not the login node.")
PY
if [[ "$cluster_dry_run" != 1 ]] && ! command -v condor_submit_dag >/dev/null; then
    echo "condor_submit_dag is unavailable; run this on the HTCondor submission host." >&2; exit 1
fi
export CLUSTER_RUN_ROOT
CLUSTER_RUN_ROOT=$("$CLUSTER_GPU_PYTHON" - "$cluster_script_dir" "$cluster_data" "$cluster_passes" \
    "$cluster_chunk_size" "$cluster_seeds" "$cluster_max_jobs" "$cluster_dry_run" \
    "$cluster_teacher_candidates" "$cluster_teacher_max_steps" "$cluster_nested_teacher" \
    "$cluster_cluster_attempts" "$cluster_validation_seconds" <<'PY'
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import tempfile

folder = Path(sys.argv[1])
scaled = bool(sys.argv[2])
source = Path(sys.argv[2]) if scaled else folder / "results/teacher_branches_20261023/development.json"
source = source.resolve()
passes, chunk_size, seeds, max_jobs = map(int, sys.argv[3:7])
teacher_candidates, nested_teacher = int(sys.argv[8]), sys.argv[10] == "1"
largest = 1 << (max(32, teacher_candidates) - 1).bit_length()
teacher_max_steps = (256 * largest if nested_teacher else 8192) if sys.argv[9] == "auto" else int(sys.argv[9])
cluster_attempts, validation_seconds = map(int, sys.argv[11:13])
teacher_stages = ([dict(candidates=2 ** power, max_steps=max(1, teacher_max_steps * 2 ** power // largest))
                   for power in range(5, largest.bit_length())] if nested_teacher else
                  [dict(candidates=teacher_candidates, max_steps=teacher_max_steps)])
parent = None if scaled else folder / "results/useful_clusters_20261021/policy_goal_actor_seed0.best.pt"
required = (source, folder / "teacher_branches_gpu.dag", folder / "teacher_branches_gpu.sub")
for path in required + ((parent,) if parent is not None else ()):
    if not path.is_file():
        raise SystemExit(f"Required shared file is missing: {path}")
metadata = json.loads(source.read_text()).get("scale_preparation", {}) if scaled else {}
if scaled:
    count = metadata.get("unresolved_train_puzzles")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise SystemExit("Scaled --data must record a positive scale_preparation.unresolved_train_puzzles count.")
    chunks = (count + chunk_size - 1) // chunk_size
    epochs = passes * chunks
prefix = "teacher_forks_scaled_" if scaled else "teacher_forks_"
run = Path(tempfile.mkdtemp(prefix=prefix + date.today().strftime("%Y%m%d") + ".", dir=folder / "results"))
shutil.copy2(source, run / "development.json")
shutil.copy2(folder / "teacher_branches_gpu.sub", run / "teacher_branches_gpu.sub")
plan = dict(mode="scaled" if scaled else "legacy", source_data=str(source),
    source_sha256=sha256(source.read_bytes()).hexdigest(), initial_checkpoint=str(parent) if parent else None,
    initial_checkpoint_sha256=sha256(parent.read_bytes()).hexdigest() if parent else None,
    initialization="seeded_random" if scaled else "checkpoint", python=sys.executable,
    run_root=str(run), max_jobs=max_jobs, dry_run=sys.argv[7] == "1",
    teacher_candidates=teacher_candidates, teacher_max_steps=teacher_max_steps, nested_teacher=nested_teacher,
    teacher_search_stages=teacher_stages, teacher_search_steps_nominal=sum(stage["max_steps"] for stage in teacher_stages),
    teacher_probes=80, teacher_mutations=32,
    cluster_attempts=cluster_attempts, validation_seconds=validation_seconds)
if scaled:
    common = dict(epochs=str(epochs), puzzles_per_epoch=str(chunk_size), validation_interval="4",
                  checkpoint_every="5", progress_every="100", job_memory="32GB",
                  teacher_candidates=str(teacher_candidates), teacher_max_steps=str(teacher_max_steps),
                  teacher_search_args="--nested-teacher" if nested_teacher else "",
                  cluster_attempts=str(cluster_attempts), validation_seconds=str(validation_seconds))
    actor = "--objective pruning --target-first --no-curriculum --no-total-work-limit --cycle-puzzles --teacher-cache-size 0"
    nodes = [("WARMUP", "warmup", 0, "learn", None, actor + f" --warmup {epochs}")]
    for phase in ("teacher", "reinforce", "q"):
        for seed in range(seeds):
            arguments = ("--batch-size 8 --updates-per-state 1 --replay-capacity 16384 --n-step 8 --target-update 100 --cycle-puzzles"
                         if phase == "q" else actor + f" --warmup {epochs if phase == 'teacher' else 0}")
            nodes.append((phase.upper() + str(seed), phase + "_seed" + str(seed), seed,
                          "qlearn" if phase == "q" else "learn", str(run / "warmup/policy.best.pt"), arguments))
    lines = ["# Generated scaled training: each dataset pass covers every unresolved training puzzle."]
    jobs = []
    for name, run_name, seed, learner, checkpoint, arguments in nodes:
        variables = dict(common, run_name=run_name, run_seed=str(seed), learner=learner,
                         fresh_start="true" if checkpoint is None else "false", phase_args=arguments)
        if checkpoint is not None:
            variables["initial_checkpoint"] = checkpoint
        escaped = lambda value: value.replace("\\", "\\\\").replace('"', '\\"')
        lines.extend((f"JOB {name} teacher_branches_gpu.sub",
                      "VARS " + name + " " + " ".join(f'{key}="{escaped(value)}"' for key, value in variables.items())))
        jobs.append(dict(node=name, run_name=run_name, seed=seed, learner=learner,
                         initial_checkpoint=checkpoint, fresh_start=checkpoint is None,
                         epochs=epochs, phase_args=arguments, teacher_candidates=teacher_candidates,
                         teacher_max_steps=teacher_max_steps, nested_teacher=nested_teacher,
                         cluster_attempts=cluster_attempts, validation_seconds=validation_seconds))
    lines.append("PARENT WARMUP CHILD " + " ".join(row[0] for row in nodes[1:]))
    (run / "teacher_branches_gpu.dag").write_text("\n".join(lines) + "\n")
    plan.update(unresolved_train_puzzles=count, passes_per_phase=passes, chunk_size=chunk_size,
        chunks_per_pass=chunks, epochs_per_phase=epochs, puzzle_visits_per_phase=count * passes,
        seeds=list(range(seeds)), jobs=jobs, common_settings=common,
        actor_teacher_cache_size=0, validation_interval=4, checkpoint_every=5,
        q_batch_size=8, q_updates_per_state=1,
        comparison="Same puzzle passes and inference caps; actor and Q optimization budgets differ.")
else:
    shutil.copy2(folder / "teacher_branches_gpu.dag", run / "teacher_branches_gpu.dag")
    plan.update(seeds=[0, 1, 2], warmup_epochs=30, continuation_epochs=60,
                puzzles_per_epoch=90, validation_interval=10, checkpoint_every=5)
(run / "run_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
print(run)
PY
)
printf 'Run directory: %s\n' "$CLUSTER_RUN_ROOT"
printf 'Run plan: %s/run_plan.json\n' "$CLUSTER_RUN_ROOT"
printf 'Monitor: tail -f "%s/warmup.out"\n' "$CLUSTER_RUN_ROOT"
if [[ "$cluster_dry_run" == 1 ]]; then
    echo "Dry run: files rendered; no job submitted."
    exit 0
fi
cd -- "$CLUSTER_RUN_ROOT"
condor_submit_dag -maxjobs "$cluster_max_jobs" \
    -include_env CLUSTER_REPO_ROOT,CLUSTER_GPU_PYTHON,CLUSTER_RUN_ROOT,PATH,LD_LIBRARY_PATH \
    teacher_branches_gpu.dag
