"""Compare interleaved construction policies with frozen individual policies."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path
import platform
import shutil
from time import perf_counter

import numpy as np
import torch

from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from .. import runtime
from .experiment import instances, solve


context = {}
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE = Path(__file__).parent / "results/completed_branches_test_20261102.1d29wvmo"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def local_path(path):
    path = Path(path)
    return path if path.exists() else ROOT / str(path).split("tnreason-py/", 1)[1]


def initialize(folder):
    torch.set_num_threads(1)
    plan = json.loads((Path(folder) / "protocol.json").read_text())
    data = json.loads(Path(plan["test"]["path"]).read_text())
    domains, _ = instances(data)
    raw = data["network"]
    network = ConstraintNetwork({core["key"]: NumpyCore(
        values=np.asarray(core["values"], dtype=bool).reshape(core["shape"]),
        colors=[raw["colors"][i] for i in core["colors"]]) for core in raw["cores"]},
        maintain_indices=True)
    models = {name: runtime.load(info["path"]) for name, info in plan["checkpoints"].items()}
    context.update(plan=plan, network=network, models=models,
                   domains=dict(zip(plan["test"]["indices"], domains)))


def evaluate(task):
    method, allowance, index, seconds = task
    plan, models = context["plan"], context["models"]
    model = (runtime.ConstructionEnsemble(models) if method == "ensemble"
             else models["teacher" if method == "heuristic" else method])
    contributions, root_trace = {}, []
    original_apply, original_heuristic = runtime.apply_candidate, runtime.attach_heuristic

    def observed(state, candidate, prepared=None, **limits):
        source = model.last_source if method == "ensemble" else method
        root = state.decision_level == 0
        before = {v: len(state.domains[v]) for v in model.reward_colors} if root else {}
        result = original_apply(state, candidate, prepared, **limits)
        counts = contributions.setdefault(source, dict(contractions=0, useful=0, removed_values=0))
        counts["contractions"] += 1
        counts["useful"] += int(result.removed_values > 0 or result.contradiction)
        counts["removed_values"] += result.removed_values
        if root:
            root_trace.append(dict(source=source, cores=candidate.core_keys,
                                   outputs=candidate.open_colors, removed_values=result.removed_values,
                                   primary_removed=sum(n - len(state.domains[v]) for v, n in before.items()),
                                   contradiction=bool(result.contradiction)))
        return result

    def heuristic(state, **options):
        options["max_steps"] = max(8192, 256 * allowance)
        return original_heuristic(state, **options)

    runtime.apply_candidate, runtime.attach_heuristic = observed, heuristic
    try:
        row = solve(context["network"], context["domains"][index],
                    "heuristic" if method == "heuristic" else "trained", model,
                    nodes=plan["nodes"], seconds=seconds, total_work=None,
                    baseline_candidates=allowance, cluster_attempts=allowance)
    finally:
        runtime.apply_candidate, runtime.attach_heuristic = original_apply, original_heuristic
    row.pop("assignment", None)  # solve() checks the assignment against all original cores.
    row.update(type="solve", method=method, allowance=allowance, index=index,
               wall_budget_s=seconds, contributions=contributions, root_trace=root_trace)
    if row["status"] in ("invalid", "unsat"):
        raise AssertionError(row)
    return row


def summarize(rows):
    complete = all(row["status"] == "solved" and row["valid"] for row in rows)
    return dict(instances=len(rows), completed=sum(row["valid"] for row in rows),
                nodes=sum(row["nodes"] for row in rows) if complete else None,
                nodes_by_puzzle={row["index"]: row["nodes"] for row in rows},
                without_branching=[row["index"] for row in rows if row["valid"] and row["nodes"] == 1],
                contractions=sum(row["work"]["contractions"] for row in rows),
                estimated_work=sum(row["work"]["estimated_work"] for row in rows),
                seconds=sum(row["solve_s"] for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, required=True, help="New result directory; must not exist")
    parser.add_argument("--methods", nargs="+", default=["ensemble", "teacher", "reinforce", "q", "heuristic"],
                        choices=("ensemble", "teacher", "reinforce", "q", "heuristic"))
    parser.add_argument("--allowances", nargs="+", type=int, default=[32, 96])
    parser.add_argument("--indices", nargs="+", type=int, help="Original test identities, e.g. 56")
    parser.add_argument("--reuse-reference32", action="store_true",
                        help="Reuse archived individual 32-attempt runs; ensembles are always measured")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--retry-seconds", type=float, default=3600)
    args = parser.parse_args()
    if min(args.allowances) < 1 or args.workers < 1 or min(args.seconds, args.retry_seconds) <= 0:
        parser.error("Allowances, workers and time guards must be positive")
    reference = json.loads((args.reference / "protocol.json").read_text())
    test = dict(reference["datasets"]["test"])
    test["path"] = str(local_path(test["path"]).resolve())
    if test["positions"] != list(range(len(test["indices"]))):
        parser.error("Reference test data must already contain exactly the selected test instances")
    selected = test["indices"] if args.indices is None else args.indices
    if not selected or set(selected) - set(test["indices"]) or len(set(selected)) != len(selected):
        parser.error("Test identities must be distinct members of the reference test set")
    checkpoints = {name: dict(path=str(local_path(reference["checkpoints"][name]["path"]).resolve()),
                              sha256=reference["checkpoints"][name]["sha256"])
                   for name in ("teacher", "reinforce", "q")}
    for info in [test, *checkpoints.values()]:
        if digest(info["path"]) != info["sha256"]:
            raise ValueError(f"Frozen input changed: {info['path']}")
    sources = {str(local_path(path).resolve()): digest(local_path(path))
               for path in reference["source_sha256"]}
    sources[str(Path(__file__).resolve())] = digest(__file__)
    plan = dict(reference=str(args.reference.resolve()), reference_sha256=digest(args.reference / "protocol.json"),
                test=test, indices=selected, checkpoints=checkpoints, sources=sources,
                methods=args.methods, allowances=args.allowances, nodes=100000, seconds=args.seconds,
                retry_seconds=args.retry_seconds, workers=args.workers, threads_per_worker=1,
                reuse_reference32=args.reuse_reference32, test_note=reference["test_note"],
                decoder_steps_per_restart=8192, ensemble_decoder_budget="Split across three members",
                order=["teacher", "reinforce", "q"], duplicate_contractions="Skip identical core/output sets",
                heuristic_search_steps="max(8192, 256 * allowance)", total_work=None,
                versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "protocol.json").write_text(json.dumps(plan, indent=2) + "\n")
    shutil.copy2(__file__, args.output / "ensemble_experiment.py")
    tasks = [(method, allowance, index, args.seconds)
             for allowance in args.allowances for index in selected for method in args.methods
             if not (args.reuse_reference32 and allowance == 32 and method != "ensemble")]
    if not tasks:
        parser.error("No new solves requested")
    results = {}
    started = perf_counter()
    with (args.output / "measurements.jsonl").open("x") as stream, ProcessPoolExecutor(
            max_workers=args.workers, mp_context=get_context("spawn"),
            initializer=initialize, initargs=(str(args.output.resolve()),)) as pool:
        for seconds in (args.seconds, args.retry_seconds):
            futures = {pool.submit(evaluate, task): task for task in tasks}
            tasks = []
            for future in as_completed(futures):
                row = future.result()
                key = (row["method"], row["allowance"], row["index"])
                results[key] = row
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(*key, row["status"], row["nodes"], f"{row['solve_s']:.2f}s",
                      f"completed={len(results)} elapsed={perf_counter()-started:.1f}s", flush=True)
                if row["status"] == "wall_timeout" and seconds < args.retry_seconds:
                    tasks.append((*key, args.retry_seconds))
            if not tasks:
                break
    summary = dict(test_note=plan["test_note"], results={})
    for method, allowance in sorted({key[:2] for key in results}):
        rows = [results[method, allowance, index] for index in selected]
        summary["results"][f"{method}_{allowance}"] = summarize(rows)
    if args.reuse_reference32:
        archived = json.loads((args.reference / "summary.json").read_text())
        summary["reference32"] = archived["results"]["test"]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for path, expected in sources.items():
        if digest(path) != expected:
            raise AssertionError(f"Source changed during evaluation: {path}")
    for info in [test, *checkpoints.values()]:
        if digest(info["path"]) != info["sha256"]:
            raise AssertionError(f"Frozen input changed during evaluation: {info['path']}")
    print("COMPLETE", json.dumps({name: row["nodes"] for name, row in summary["results"].items()}), flush=True)


if __name__ == "__main__":
    main()
