"""Explicit Nikoli100 experiment: plain MAC versus Sudoku cluster contractions.

Run from tnreason-py with ``python -m experiments.sudoku_rule_clusters``.
No solver defaults or benchmark adapters are changed by this experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import signal
from time import perf_counter
import traceback

from demonstrations.sudoku.examples.standard_sudoku import get_assignment_as_constraint_network
from demonstrations.sudoku.inferenceClusters import (
    RuleContext, RuleLimits, RuleRegistry, available_rules, contract_cluster,
)
from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.mac import ConstraintNetworkSolver


class ClusterClosure:
    """Apply exact unary cluster projections through MAC's reversible domains."""

    def __init__(self, network, families, limits):
        self.network, self.limits = network, limits
        self.registries = {name: RuleRegistry((name,)) for name in families}
        self.counts = Counter()
        self.by_family = {name: Counter() for name in families}
        self.last = None

    def __call__(self, state, changed):
        if state.contradiction or state.is_solved():
            return {}
        snapshot = tuple(state.domains.items())
        if snapshot == self.last:
            return {}
        self.last = snapshot
        self.counts["closures"] += 1
        started = perf_counter()
        try:
            context = RuleContext(self.network, state.domains, num=3, limits=self.limits)
            # This context is an immutable domain snapshot, so these local
            # option views can be reused safely during its detectors.
            context.options = lru_cache(maxsize=None)(context.options)
            seen = set()
            for family, registry in self.registries.items():
                counters = self.by_family[family]
                detection = registry.detect(context)[family]
                counters["scans"] += 1
                counters["detection_steps"] += detection.steps
                counters["truncated_scans"] += int(detection.truncated)
                counters["detected"] += len(detection.clusters)
                for cluster in detection.clusters:
                    # Singles at GAC closure already have their fixed support.
                    if family == "singles":
                        counters["already_gac"] += 1
                        continue
                    signature = cluster.core_keys, cluster.open_colors
                    if signature in seen:
                        counters["duplicate"] += 1
                        continue
                    seen.add(signature)
                    result = contract_cluster(context, cluster)
                    counters["contraction_work"] += result.work
                    counters["max_tensor_entries"] = max(counters["max_tensor_entries"], result.max_size)
                    if result.status != "executed":
                        counters["skipped_budget"] += 1
                        continue
                    counters["contracted"] += 1
                    removed = 0
                    for color, allowed in result.supported_domains.items():
                        before = len(state.domains[color])
                        state.restrict_domain(color, allowed)
                        removed += before - len(state.domains[color])
                    if result.contradiction and not state.contradiction:
                        state.restrict_domain(cluster.participating_variables[0], ())
                    counters["direct_removed_values"] += removed
                    if removed or state.contradiction:
                        counters["useful"] += 1
                        counters["contradictions"] += int(state.contradiction)
                        # Run GAC before selecting the next cluster. No
                        # domain-dependent summary is installed or cached.
                        return {}
            return {}
        finally:
            self.counts["time_s"] += perf_counter() - started

    def statistics(self):
        return {**self.counts, "by_family": {name: dict(counts) for name, counts in self.by_family.items()}}


def _timeout(signum, frame):
    raise TimeoutError("Per-puzzle wall-clock limit exceeded")


def solve_puzzle(puzzle, mode, families, limits, max_nodes, timeout):
    row = {"instance_id": puzzle["puzzle_id"], "mode": mode,
           "nodes": None, "branch_attempts": 0, "root_solved": False,
           "solved": False, "valid": False}
    started, policy, solver = perf_counter(), None, None
    previous_handler = signal.signal(signal.SIGALRM, _timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        evidence = initial_board_into_evidence(puzzle["initial_board"], colNum=3, rowNum=3)
        network = ConstraintNetwork(get_assignment_as_constraint_network(3, evidence), maintain_indices=True)
        state = network.create_mac_state()
        if mode == "clusters":
            policy = ClusterClosure(network, families, limits)
            state.on_closure = policy
        original_assign = state.assign

        def counted_assign(color, value):
            row["branch_attempts"] += 1
            return original_assign(color, value)

        state.assign = counted_assign
        solver = ConstraintNetworkSolver(network, max_nodes=max_nodes)
        root_start = perf_counter()
        state.propagate()
        row["root_time_s"] = perf_counter() - root_start
        row["root_solved"] = state.is_solved()
        row["root_unresolved_cells"] = sum(
            color.startswith("pos_") and len(domain) > 1 for color, domain in state.domains.items())
        result = solver.solve(state=state)
        assignment = result.assignment or {}
        row.update(nodes=result.nodes, solved=result.solved,
                   status="node_limit" if result.node_limit_reached else (
                       "solved" if result.solved else "unsat"),
                   mac_statistics=asdict(result.statistics))
        if result.solved:
            board = "".join(str(assignment[f"pos_{r // 3}_{r % 3}_{c // 3}_{c % 3}"] + 1)
                            for r in range(9) for c in range(9))
            valid = (board == puzzle["solution"] and all(
                digit in ".0" or board[i] == digit for i, digit in enumerate(puzzle["initial_board"])))
            valid = valid and all(core[{color: assignment[color] for color in core.colors}] != 0
                                  for core in network.coresDict.values())
            row.update(valid=bool(valid), output_board=board)
            if not valid:
                row["status"] = "invalid"
    except TimeoutError as error:
        row.update(status="timeout", error=str(error), nodes=solver.nodes if solver else None)
    except Exception:
        row.update(status="error", error=traceback.format_exc(), nodes=solver.nodes if solver else None)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        row["total_time_s"] = perf_counter() - started
        if policy is not None:
            row["cluster_statistics"] = policy.statistics()
    return row


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root.parent / "tnreason-benchmarks/data/sudoku/nikoli100.parquet")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--modes", nargs="+", choices=("plain", "clusters"), default=("plain", "clusters"))
    parser.add_argument("--families", nargs="+", choices=available_rules(), default=available_rules())
    parser.add_argument("--workloads", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-nodes", type=int, default=1_000_000)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--max-patterns", type=int, default=64)
    parser.add_argument("--max-work", type=int, default=10_000_000)
    args = parser.parse_args()
    if args.timeout <= 0 or args.max_nodes <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("timeout, max-nodes and limit must be positive")
    import pandas as pd
    puzzles = pd.read_parquet(args.dataset).to_dict("records")
    if args.workloads:
        selected = set(args.workloads)
        missing = selected - {puzzle["puzzle_id"] for puzzle in puzzles}
        if missing:
            parser.error(f"Unknown workloads: {sorted(missing)}")
        puzzles = [puzzle for puzzle in puzzles if puzzle["puzzle_id"] in selected]
    if args.limit:
        puzzles = puzzles[:args.limit]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output_dir or root / "experiments/cluster_selection/results" / f"sudoku_rules_{stamp}"
    destination.mkdir(parents=True, exist_ok=False)
    limits = RuleLimits(max_steps=args.max_steps, max_patterns=args.max_patterns, max_work=args.max_work)
    protocol = {"dataset": str(args.dataset.resolve()), "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                "instances": [puzzle["puzzle_id"] for puzzle in puzzles], "modes": args.modes,
                "families": args.families, "limits": asdict(limits), "timeout_s": args.timeout,
                "max_nodes": args.max_nodes, "nodes_include_root": True,
                "application": "exact unary cluster projections at each GAC closure; reversible domain restrictions",
                "branching": "unchanged MAC MRV-degree policy; no summary cores added"}
    (destination / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows = []
    with (destination / "measurements.jsonl").open("x") as output:
        for index, puzzle in enumerate(puzzles, 1):
            for mode in args.modes:
                row = solve_puzzle(puzzle, mode, args.families, limits, args.max_nodes, args.timeout)
                output.write(json.dumps(row) + "\n")
                output.flush()
                rows.append(row)
                print(f"{index}/{len(puzzles)} {puzzle['puzzle_id']} {mode}: {row['status']} "
                      f"nodes={row['nodes']} attempts={row['branch_attempts']} "
                      f"root={row['root_solved']} time={row['total_time_s']:.3f}s", flush=True)
    summary = {}
    for mode in args.modes:
        selected = [row for row in rows if row["mode"] == mode]
        summary[mode] = {"instances": len(selected), "solved_valid": sum(row["solved"] and row["valid"] for row in selected),
                         "status_counts": dict(Counter(row["status"] for row in selected)),
                         "total_nodes": sum(row["nodes"] or 0 for row in selected),
                         "branch_attempts": sum(row["branch_attempts"] for row in selected),
                         "root_solved": sum(row["root_solved"] for row in selected),
                         "total_time_s": sum(row["total_time_s"] for row in selected)}
    paired = {puzzle["puzzle_id"]: {row["mode"]: row for row in rows if row["instance_id"] == puzzle["puzzle_id"]}
              for puzzle in puzzles}
    comparisons = [(pair["plain"], pair["clusters"]) for pair in paired.values()
                   if "plain" in pair and "clusters" in pair and all(row["solved"] and row["valid"] for row in pair.values())]
    if comparisons:
        summary["comparison"] = {"completed_pairs": len(comparisons),
                                 "fewer_nodes": sum(cluster["nodes"] < plain["nodes"] for plain, cluster in comparisons),
                                 "equal_nodes": sum(cluster["nodes"] == plain["nodes"] for plain, cluster in comparisons),
                                 "more_nodes": sum(cluster["nodes"] > plain["nodes"] for plain, cluster in comparisons)}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results: {destination}", flush=True)
    return 0 if all(row["solved"] and row["valid"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
