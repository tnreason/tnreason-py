"""Compare propagation and full MAC solves, including untrained controls."""
import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import signal
import sys
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import torch
from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from .. import runtime
from ..candidate_generation import ClusterCandidateGenerator
from ..data import network_data
from .model import Policy


def instances(data):
    """Generic domain instances; digit-grid datasets use the existing input adapter."""
    if "initial_domains" in data:
        return data["initial_domains"], data.get("instance_ids", data["initial_domains"])
    from ..supervised.collect import sudoku_domains
    boards = data["puzzles"]
    return [sudoku_domains(board) for board in boards], boards


def fingerprints(data, domains):
    dimensions = dict(zip(data["network"]["colors"], data["network"]["dimensions"]))
    return {tuple(sorted((color, tuple(sorted(set(values)))) for color, values in instance.items()
                         if set(values) != set(range(dimensions[color])))) for instance in domains}


def instance_hashes(data, domains):
    return {sha256(json.dumps(item, separators=(",", ":")).encode()).hexdigest()
            for item in fingerprints(data, domains)}


class WorkBudgetExceeded(RuntimeError):
    pass


class First:
    """First structural proposal, using the same repeat/stop rule as construction."""
    kind = "construction"

    def __init__(self, model, *, candidates=32):
        self.network, self.config = model.network, model.config
        self.reward_colors = getattr(model, "reward_colors", self.network["colors"])
        cores = {core["key"]: SimpleNamespace(
            colors=[self.network["colors"][idx] for idx in core["colors"]], shape=core["shape"])
            for core in self.network["cores"]}
        limits = {key: self.config[key] for key in ("max_cores", "max_open_colors", "max_work", "max_size")}
        self.generator = ClusterCandidateGenerator(cores, max_candidates=candidates, **limits)

    def propose(self, domains):
        candidates = self.generator.generate(domains, cyclic_only=True,
                                             compact_domains=self.config.get("compact_domains", False))
        return candidates[0] if candidates else None


def _valid_assignment(network, domains, assignment):
    return bool(set(assignment) == set(network.dimension_dict)
                and all(assignment[color] in values for color, values in domains.items())
                and all(core[{color: assignment[color] for color in core.colors}] != 0
                        for core in network.coresDict.values()))


def _run(network, domains, mode, model, *, nodes, seconds, total_work, propagation_only,
         baseline_candidates, cluster_attempts=None, stop_on_noop=False, baseline_steps=None):
    """Both evaluations share cluster policy, work accounting, and validation."""
    if not np.isfinite(seconds) or seconds <= 0 or total_work is not None and total_work < 1:
        raise ValueError("Evaluation budgets must be positive")
    if baseline_steps is not None and (type(baseline_steps) is not int or baseline_steps < 1):
        raise ValueError("Structural search steps must be a positive integer")
    started = perf_counter()
    state = network.create_mac_state(domains)
    initial_domains = state.snapshot_domains()
    setup_s = perf_counter() - started
    work = dict(contractions=0, estimated_work=0, max_tensor_size=0, budget_exhausted=False)
    policy = None
    original = runtime.contract_candidate

    def attach():
        nonlocal policy
        if mode == "heuristic":
            limits = {key: model.config[key] for key in ("max_cores", "max_open_colors", "max_work", "max_size")}
            if baseline_steps is not None:
                limits["max_steps"] = baseline_steps
            runtime.attach_heuristic(state, max_candidates=baseline_candidates,
                                     compact_domains=model.config.get("compact_domains", False), **limits)
        elif mode != "plain":
            options = (dict(max_candidates=baseline_candidates if cluster_attempts is None else cluster_attempts,
                            stop_on_noop=stop_on_noop)
                       if getattr(model, "kind", None) == "construction" else {})
            runtime.attach(state, model, **options)
        policy = state.on_closure
        if total_work is not None and policy is not None:
            state.on_closure = budgeted

    def measured(candidate, prepared):
        estimate = prepared[0]
        if total_work is not None and work["estimated_work"] + estimate.work > total_work:
            raise WorkBudgetExceeded("Next contraction would exceed cumulative work budget")
        work["contractions"] += 1
        work["estimated_work"] += estimate.work
        work["max_tensor_size"] = max(work["max_tensor_size"], estimate.max_size)
        return original(candidate, prepared)

    def budgeted(state, changed):
        try:
            return policy(state, changed)
        except WorkBudgetExceeded:
            work["budget_exhausted"] = True
            state.on_closure = None  # Finish with ordinary MAC, retaining earlier pruning.

    def timeout(*_):
        raise TimeoutError("Search exceeded wall budget")

    runtime.contract_candidate = measured
    previous = signal.signal(signal.SIGALRM, timeout)
    started = perf_counter()
    baseline_domains = None
    baseline_statistics = None
    baseline_solved = False
    baseline_contradiction = False
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        if propagation_only:
            state.propagate()
            baseline_domains, baseline_statistics = state.snapshot_domains(), state.statistics
            baseline_solved = state.is_solved()
            baseline_contradiction = state.contradiction
            attach()
            state.propagate()
            elapsed = perf_counter() - started
            signal.setitimer(signal.ITIMER_REAL, 0)
            assignment = state.assignment() if state.is_solved() else {}
            solved = state.is_solved()
            valid = solved and _valid_assignment(network, domains, assignment)
            status = ("invalid" if solved and not valid else "solved" if solved else
                      "unsat" if state.contradiction else "work_limit" if work["budget_exhausted"] else "stalled")
            row = dict(status=status, nodes=0, solved=bool(solved and valid), valid=bool(valid),
                       assignment=assignment)
        else:
            attach()
            result = network.solve_mac(state=state, max_nodes=nodes)
            elapsed = perf_counter() - started
            signal.setitimer(signal.ITIMER_REAL, 0)
            assignment = result.assignment or {}
            valid = result.solved and _valid_assignment(network, domains, assignment)
            row = dict(status="invalid" if result.solved and not valid else "solved" if result.solved else "node_limit" if result.node_limit_reached
                       else "unsat" if result.unsatisfiable else "partial", nodes=result.nodes,
                       solved=bool(result.solved and valid), valid=bool(valid), assignment=assignment)
    except TimeoutError:
        elapsed = perf_counter() - started
        row = dict(status="wall_timeout", nodes=0 if propagation_only else None, solved=False, valid=False)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        runtime.contract_candidate = original
    row.update(setup_s=setup_s, solve_s=elapsed, work=work, mac_statistics=asdict(state.statistics),
               cluster_statistics=dict(policy.statistics) if policy else {})
    if propagation_only:
        colors = getattr(model, "reward_colors", tuple(network.dimension_dict))
        final_domains = state.snapshot_domains()
        entropy = lambda ds: sum(np.log(len(ds[color])) for color in colors) if all(ds[color] for color in colors) else None
        initial_entropy = entropy(initial_domains)
        final_entropy = None if state.contradiction else entropy(final_domains)
        baseline_entropy = entropy(baseline_domains) if baseline_domains is not None else None
        row.update(domains=final_domains, initial_entropy=initial_entropy, remaining_entropy=final_entropy,
                   baseline_entropy=baseline_entropy,
                   baseline_solved=baseline_solved, baseline_contradiction=baseline_contradiction,
                   resolved=row["solved"] or row["status"] == "unsat",
                   progress=None if final_entropy is None or baseline_entropy is None else
                   1.0 - final_entropy / baseline_entropy if baseline_entropy else float(row["solved"]),
                   cluster_removed_values=state.statistics.removed_values - baseline_statistics.removed_values
                   if baseline_statistics is not None else 0)
    if state._cores.keys() != network.coresDict.keys() or state.decision_level:
        raise AssertionError("Search left temporary cores or decision levels")
    return row


def solve(network, domains, mode, model, *, nodes, seconds, total_work=None, baseline_candidates=32,
          cluster_attempts=None, stop_on_noop=False, baseline_steps=None):
    """Run DFS with temporary cluster summaries and validate any full solution."""
    return _run(network, domains, mode, model, nodes=nodes, seconds=seconds, total_work=total_work,
                propagation_only=False, baseline_candidates=baseline_candidates,
                cluster_attempts=cluster_attempts, stop_on_noop=stop_on_noop, baseline_steps=baseline_steps)


def propagate(network, domains, mode, model, *, seconds, total_work=None, baseline_candidates=32,
              cluster_attempts=None, stop_on_noop=False, baseline_steps=None):
    """Close ordinary GAC, then cluster/GAC repeatedly, with no decisions or DFS."""
    return _run(network, domains, mode, model, nodes=0, seconds=seconds, total_work=total_work,
                propagation_only=True, baseline_candidates=baseline_candidates,
                cluster_attempts=cluster_attempts, stop_on_noop=stop_on_noop, baseline_steps=baseline_steps)


def evaluate_roots(network, initial_domains, model, *, seconds, total_work=None, mode="trained", cluster_attempts=None):
    """Checkpoint metrics on complete roots; contradictory roots never count as solved."""
    rows = [propagate(network, domains, mode, model, seconds=seconds, total_work=total_work,
                      cluster_attempts=cluster_attempts)
            for domains in initial_domains]
    unresolved = [row for row in rows if not row["baseline_solved"] and not row["baseline_contradiction"]]
    return dict(instances=len(rows), solved=sum(row["solved"] for row in rows),
                solved_fraction=sum(row["solved"] for row in rows) / max(1, len(rows)),
                baseline_solved=sum(row["baseline_solved"] for row in rows),
                baseline_contradictions=sum(row["baseline_contradiction"] for row in rows),
                resolved=sum(row["resolved"] for row in rows),
                resolved_fraction=sum(row["resolved"] for row in rows) / max(1, len(rows)),
                unresolved=len(unresolved), additional_solved=sum(row["solved"] for row in unresolved),
                additional_solved_fraction=sum(row["solved"] for row in unresolved) / max(1, len(unresolved)),
                additional_resolved=sum(row["resolved"] for row in unresolved),
                additional_resolved_fraction=sum(row["resolved"] for row in unresolved) / max(1, len(unresolved)),
                mean_progress=float(np.mean([row["progress"] or 0.0 for row in unresolved])) if unresolved else 0.0,
                contradictions=sum(row["status"] == "unsat" for row in rows),
                mean_seconds=sum(row["solve_s"] for row in rows) / max(1, len(rows)), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Evaluation dataset with a raw network and instances")
    parser.add_argument("--training-data", type=Path, required=True, help="Reject any exact instance overlap")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSONL file; never overwritten")
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--modes", nargs="+", choices=("plain", "trained", "untrained", "prior", "first", "heuristic"),
                        default=("plain", "trained", "untrained", "prior", "first", "heuristic"))
    parser.add_argument("--nodes", type=int, default=200)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--total-work", type=int, help="Cumulative contraction work per solve; then finish with ordinary MAC")
    parser.add_argument("--propagation-only", action="store_true", help="Run repeated cluster/GAC inference without DFS")
    parser.add_argument("--baseline-candidates", type=int, default=32,
                        help="Structural baseline proposals, independent of policy training proposals")
    parser.add_argument("--cluster-attempts", type=int,
                        help="Learned affordable contractions per closure; defaults to --baseline-candidates")
    parser.add_argument("--stop-on-noop", action="store_true",
                        help="Reproduce legacy greedy construction, stopping at its first no-op")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, help="Override checkpoint prior strength in every construction mode")
    parser.add_argument("--decoder", choices=("depth_first", "discrepancy", "sampling", "sampling_grow"),
                        help="Override construction decoding; sampling_grow extends legal additions (Q sampling is experimental)")
    args = parser.parse_args()
    if (args.output.exists() or args.nodes < 1 or args.baseline_candidates < 1
            or args.cluster_attempts is not None and args.cluster_attempts < 1
            or not np.isfinite(args.seconds) or args.seconds <= 0
            or args.total_work is not None and args.total_work < 1):
        parser.error("Output must be new and budgets must be positive")
    data = json.loads(args.data.read_text())
    training = json.loads(args.training_data.read_text())
    domains, identities = instances(data)
    training_domains, _ = instances(training)
    if any(index < 0 or index >= len(domains) for index in args.indices):
        parser.error("Instance index outside evaluation dataset")
    overlap = fingerprints(data, [domains[index] for index in args.indices]) & fingerprints(training, training_domains)
    if overlap:
        parser.error(f"Selected evaluation instances overlap training data in {len(overlap)} instances")
    raw = data["network"]
    network = ConstraintNetwork({core["key"]: NumpyCore(
        values=np.asarray(core["values"], dtype=bool).reshape(core["shape"]),
        colors=[raw["colors"][idx] for idx in core["colors"]]) for core in raw["cores"]}, maintain_indices=True)
    torch.set_num_threads(1)
    started = perf_counter()
    trained = runtime.load(args.checkpoint)
    if getattr(trained, "kind", None) != "construction" or trained.network != network_data(network):
        parser.error("Expected a construction checkpoint matching the evaluation network")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    exposure = saved.get("training", {}).get("train_instance_hashes")
    checkpoint_overlap = set(exposure or ()) & instance_hashes(data, [domains[idx] for idx in args.indices])
    if checkpoint_overlap:
        parser.error(f"Selected evaluation instances overlap checkpoint training in {len(checkpoint_overlap)} instances")
    if args.beta is not None:
        if not np.isfinite(args.beta) or args.beta < 0:
            parser.error("Beta must be finite and nonnegative")
        trained.beta = args.beta
    if args.decoder is not None:
        trained.config["decoder"] = args.decoder
    torch.manual_seed(args.seed)
    untrained = Policy(raw, **trained.config).eval()
    untrained.reward_colors, untrained.beta = trained.reward_colors, trained.beta
    prior = deepcopy(untrained)
    with torch.no_grad():
        prior.head[-1].weight.zero_()
        prior.head[-1].bias.zero_()
    models = dict(plain=trained, trained=trained, untrained=untrained, prior=prior,
                  first=First(trained, candidates=args.baseline_candidates), heuristic=trained)
    load_s = perf_counter() - started
    sources = [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("train.py"),
               Path(runtime.__file__), *(Path(runtime.__file__).with_name(name) for name in
               ("candidate_evaluation_mac.py", "candidate_generation.py", "contraction_cost.py")),
               Path(__file__).parents[2] / "constraint_networks" / "mac.py"]
    hashes = {str(path): sha256(path.read_bytes()).hexdigest()
              for path in (args.data, args.training_data, args.checkpoint, *sources)}
    manifest = dict(type="manifest", version=4, indices=args.indices, modes=args.modes, seed=args.seed,
                    inference="propagation" if args.propagation_only else "dfs", baseline_candidates=args.baseline_candidates,
                    cluster_attempts=args.baseline_candidates if args.cluster_attempts is None else args.cluster_attempts,
                    stop_on_noop=args.stop_on_noop,
                    node_budget=args.nodes, wall_budget_s=args.seconds, total_work_budget=args.total_work,
                    threads=1, config=trained.config, command=[sys.executable, *sys.argv],
                    beta=trained.beta, load_s=load_s, exact_training_overlap=len(overlap), hashes=hashes,
                    checkpoint_training_overlap=len(checkpoint_overlap),
                    checkpoint_training_fingerprints_available=exposure is not None,
                    note="Construction and structural batch have equal default proposal allowances; consumed work can differ. "
                         "Work-budget exhaustion disables clusters and continues ordinary MAC. "
                         "Exact disjointness does not exclude transformed duplicates. Timings exclude validation.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(manifest) + "\n")
        stream.flush()
        for index in args.indices:
            for mode in args.modes:
                options = dict(seconds=args.seconds, total_work=args.total_work,
                               baseline_candidates=args.baseline_candidates,
                               cluster_attempts=args.cluster_attempts, stop_on_noop=args.stop_on_noop)
                row = (propagate(network, domains[index], mode, models[mode], **options) if args.propagation_only else
                       solve(network, domains[index], mode, models[mode], nodes=args.nodes, **options))
                assignment = row.pop("assignment", {})
                row.update(type="propagate" if args.propagation_only else "solve", index=index, instance=identities[index], mode=mode,
                           node_budget=args.nodes, wall_budget_s=args.seconds, total_work_budget=args.total_work,
                           targets={color: assignment[color] for color in data.get("reward_colors", ()) if color in assignment})
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(index, mode, row["status"], row["nodes"], f"{row['solve_s']:.3f}s", flush=True)


if __name__ == "__main__":
    main()
