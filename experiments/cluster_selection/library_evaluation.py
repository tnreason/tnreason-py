"""Exact, independent coverage measurements for a frozen cluster library.

Construct one LibraryEvaluator per worker and reuse it across closed states.
Dense plans retain live domain tensors; summaries and trial outcomes are never
cached. Sparse successes include auxiliary pruning, not only primary pruning.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite, log
from pathlib import Path
import platform
import signal
from time import perf_counter

import numpy as np
from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.mac import ConstraintNetworkSolver
from .candidate_evaluation_mac import prepare_candidate, contract_candidate
from .candidate_evaluation_tnreason import SummaryClusterCandidate
from .candidate_generation import ClusterCandidateGenerator
from .data import network_data
from .runtime import apply_candidate
from .library import _write


def prefix_metrics(successes, sizes):
    """Oracle coverage of nested library prefixes, without fitting a ranker."""
    return [dict(size=size, useful=bool(rows := [row for row in successes if row["index"] < size]),
                 successes=len(rows), best_gain=max((row["gain"] for row in rows), default=0.),
                 best_primary_gain=max((row["primary_gain"] for row in rows), default=0.))
            for size in sizes]


class LibraryEvaluator:
    def __init__(self, library, *, max_work=1_000_000, max_size=100_000):
        if any(type(value) is not int or value < 1 for value in (max_work, max_size)):
            raise ValueError("Expected positive contraction limits")
        raw = library["network"]
        self.colors, keys = raw["colors"], [core["key"] for core in raw["cores"]]
        network = ConstraintNetwork({core["key"]: NumpyCore(
            values=np.asarray(core["values"], dtype=bool).reshape(core["shape"]),
            colors=[self.colors[i] for i in core["colors"]]) for core in raw["cores"]},
            maintain_indices=True)
        self.primary = set(library.get("reward_colors", self.colors))
        if network_data(network) != raw or not self.primary <= set(self.colors):
            raise ValueError("Invalid network schema or reward colors")
        self.state = network.create_mac_state()
        if not self.state.propagate():
            raise ValueError("Expected a noncontradictory base network")
        auxiliary = min(1., len(self.primary) / max(1, len(self.colors) - len(self.primary)))
        self.weights = {color: 1. if color in self.primary else auxiliary for color in self.colors}
        self.limits, self.plans, self.candidates = dict(max_work=max_work, max_size=max_size), {}, []
        seen = set()
        for index, entry in enumerate(library["clusters"]):
            cores, outputs = entry["cores"], entry["open"]
            if (not cores or any(type(i) is not int or not 0 <= i < len(keys) for i in cores)
                    or any(type(i) is not int or not 0 <= i < len(self.colors) for i in outputs)
                    or len(set(cores)) != len(cores) or len(set(outputs)) != len(outputs)):
                raise ValueError("Invalid library cluster indices")
            candidate = SummaryClusterCandidate("library_" + str(index),
                tuple(keys[i] for i in cores), tuple(self.colors[i] for i in outputs))
            scope = {color for key in candidate.core_keys for color in self.state._cores[key].colors}
            signature = frozenset(cores), frozenset(outputs)
            if not set(candidate.open_colors) <= scope or signature in seen:
                raise ValueError("Duplicate cluster or outputs outside its cores")
            seen.add(signature)
            self.candidates.append(candidate)
        self.network = network

    def _plan(self, index):
        if index not in self.plans:
            try:
                self.plans[index] = prepare_candidate(self.state, self.candidates[index], **self.limits)
            except ValueError as error:
                if str(error) not in ("Candidate exceeds the contraction work or tensor-size budget",
                                      "NumPy contraction supports at most 52 distinct colors"):
                    raise
                self.plans[index] = None
        return self.plans[index]

    @contextmanager
    def closed(self, record):
        """Load a complete recorded GAC closure and restore the base afterward."""
        if "masks" in record:
            masks = record["masks"]
            if len(masks) != len(self.colors) or any(type(mask) is not int or not
                    0 < mask < 1 << self.state.dimensions[color] for color, mask in zip(self.colors, masks)):
                raise ValueError("Invalid full-domain masks")
            domains = {color: tuple(value for value in range(self.state.dimensions[color])
                       if mask & 1 << value) for color, mask in zip(self.colors, masks)}
        else:
            values = record["domains"]
            if len(values) != len(self.colors):
                raise ValueError("Expected every variable's domain")
            domains = {color: tuple(values[color] if isinstance(values, dict) else values[i])
                       for i, color in enumerate(self.colors)}
        self.state.push_level()
        counters = vars(self.state.statistics)
        try:
            for color, values in domains.items():
                self.state.restrict_domain(color, values)
            if not self.state.propagate() or self.state.snapshot_domains() != domains:
                raise ValueError("Expected a noncontradictory GAC-closed state")
            yield self.state
        finally:
            self.state.pop_level()
            for name, value in counters.items():
                setattr(self.state, "_" + name, value)

    def measure(self, candidate, prepared=None, normalizers=None):
        """Measure one reversible trial on self.state; None means unaffordable."""
        state = self.state
        if state.contradiction or state._queue or state._queued or state.on_closure is not None:
            raise ValueError("Measure only a closed state without a closure hook")
        if prepared is None:
            try:
                prepared = prepare_candidate(state, candidate, **self.limits)
            except ValueError as error:
                if str(error) not in ("Candidate exceeds the contraction work or tensor-size budget",
                                      "NumPy contraction supports at most 52 distinct colors"):
                    raise
                return None
        summary = contract_candidate(candidate, prepared)
        values = summary.values
        supported = bool(values) if not candidate.open_colors else all(
            np.all(np.any(values, axis=tuple(j for j in range(values.ndim) if j != axis))
                   [list(state.domains[color])]) for axis, color in enumerate(candidate.open_colors))
        if supported:
            # Adding a core with full current-domain marginals cannot change a
            # GAC closure, even when that core contains nontrivial correlations.
            return dict(gain=0., primary_gain=0., removed_values=0,
                primary_removed_values=0, contradiction=False, work=prepared[0].work,
                oracle_contractions=1)
        remaining, primary_remaining = normalizers or (
            sum(self.weights[c] * log(len(values)) for c, values in state.domains.items()),
            sum(log(len(state.domains[c])) for c in self.primary))
        counters, mark = vars(state.statistics), len(state._trail)
        state.push_level()
        try:
            result = apply_candidate(state, candidate, prepared)
            before = {}
            for color, old_values in state._trail[mark:]:
                before.setdefault(color, old_values)
            removed = {color: len(values) - len(state.domains[color]) for color, values in before.items()}
            information = primary_information = 0.
            if not result.contradiction:
                changes = {c: log(len(values) / len(state.domains[c])) for c, values in before.items()}
                information = sum(self.weights[c] * value for c, value in changes.items())
                primary_information = sum(value for c, value in changes.items() if c in self.primary)
            return dict(gain=1. if result.contradiction else information / remaining if remaining else 0.,
                primary_gain=1. if result.contradiction else
                    primary_information / primary_remaining if primary_remaining else 0.,
                removed_values=sum(removed.values()),
                primary_removed_values=sum(value for c, value in removed.items() if c in self.primary),
                contradiction=bool(result.contradiction), work=prepared[0].work,
                oracle_contractions=2)
        finally:
            state.pop_level()
            for name, value in counters.items():
                setattr(state, "_" + name, value)

    def score_state(self, record, *, sizes=(), reference_candidates=0, reference_steps=None):
        """Return sparse successes, rejected indices and optional structural oracle."""
        sizes = sorted(set(sizes or (len(self.candidates),)))
        if any(type(size) is not int or not 0 <= size <= len(self.candidates) for size in sizes):
            raise ValueError("Prefix sizes must lie within the library")
        started, successes, rejected, contractions = perf_counter(), [], [], 0
        with self.closed(record) as state:
            normalizers = (sum(self.weights[c] * log(len(v)) for c, v in state.domains.items()),
                           sum(log(len(state.domains[c])) for c in self.primary))
            for index, candidate in enumerate(self.candidates):
                if (prepared := self._plan(index)) is None:
                    rejected.append(index)
                    continue
                outcome = self.measure(candidate, prepared, normalizers)
                contractions += outcome["oracle_contractions"]
                if outcome["removed_values"] or outcome["contradiction"]:
                    successes.append(dict(index=index, **outcome))
            reference = None
            if reference_candidates:
                generator = ClusterCandidateGenerator(self.network, max_candidates=reference_candidates,
                    max_cores=64, max_open_colors=6, max_steps=reference_steps or max(8192, 256 * reference_candidates),
                    **self.limits)
                proposals = generator.generate(state.domains, cyclic_only=True)
                outcomes = [outcome for candidate in proposals
                            if (outcome := self.measure(candidate, normalizers=normalizers)) is not None]
                reference = dict(proposed=len(proposals), evaluated=len(outcomes),
                    oracle_contractions=sum(row["oracle_contractions"] for row in outcomes),
                    useful=sum(bool(row["removed_values"] or row["contradiction"]) for row in outcomes),
                    best_gain=max((row["gain"] for row in outcomes), default=0.),
                    best_primary_gain=max((row["primary_gain"] for row in outcomes), default=0.))
        return dict(**{key: record[key] for key in ("id", "puzzle", "kind", "level") if key in record},
            evaluated=len(self.candidates) - len(rejected), rejected=rejected, successes=successes,
            oracle_contractions=contractions, prefixes=prefix_metrics(successes, sizes),
            reference=reference, seconds=perf_counter() - started)


def solve_library(library, domains, *, allowance=64, nodes=100_000, seconds=600.,
                  max_work=1_000_000, max_size=100_000):
    """Expensive greedy-pruning oracle: this measures library potential, not speed.

    At each GAC closure measure all library entries independently, apply the
    best useful one, and repeat on updated domains. Plans alone are reused.
    """
    if (any(type(value) is not int or value < 1 for value in (allowance, nodes))
            or not isfinite(seconds) or seconds <= 0):
        raise ValueError("Expected positive application, node and time limits")
    started = perf_counter()
    evaluator = LibraryEvaluator(library, max_work=max_work, max_size=max_size)
    state, counters = evaluator.state, dict(calls=0, accepted_applications=0,
        oracle_measurements=0, oracle_contractions=0, application_contractions=0, rejected=0)
    for color, values in domains.items():
        state.restrict_domain(color, values)

    def choose(current, changed):
        if current.contradiction or current.is_solved():
            return
        hook, current.on_closure = current.on_closure, None
        counters["calls"] += 1
        try:
            for _ in range(allowance):
                normalizers = (sum(evaluator.weights[c] * log(len(v)) for c, v in current.domains.items()),
                               sum(log(len(current.domains[c])) for c in evaluator.primary))
                best = None
                for index, candidate in enumerate(evaluator.candidates):
                    if (prepared := evaluator._plan(index)) is None:
                        counters["rejected"] += 1
                        continue
                    outcome = evaluator.measure(candidate, prepared, normalizers)
                    counters["oracle_measurements"] += 1
                    counters["oracle_contractions"] += outcome["oracle_contractions"]
                    rank = (-outcome["gain"], prepared[0].work, index)
                    if (outcome["removed_values"] or outcome["contradiction"]) and (best is None or rank < best[0]):
                        best = rank, candidate, prepared
                if best is None:
                    break
                result = apply_candidate(current, best[1], best[2])
                counters["accepted_applications"] += 1
                counters["application_contractions"] += 1
                if not (result.removed_values or result.contradiction):
                    raise AssertionError("Oracle-selected useful cluster lost its measured pruning")
                if current.contradiction or current.is_solved():
                    break
        finally:
            current.on_closure = hook

    def deadline(*_):
        raise TimeoutError("Fixed-library oracle solving deadline")

    solver, result = ConstraintNetworkSolver(evaluator.network, max_nodes=nodes), None
    previous = signal.signal(signal.SIGALRM, deadline)
    state.on_closure = choose
    try:
        signal.setitimer(signal.ITIMER_REAL, max(.000001, seconds - (perf_counter() - started)))
        result = solver.solve(state=state)
        status = "solved" if result.solved else "unsat" if result.unsatisfiable else "node_limit"
    except TimeoutError:
        status = "time_limit"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        state.on_closure = None
    if state.decision_level or state._temporary_core_keys or state._added_core_keys:
        raise AssertionError("Oracle solving left reversible levels or temporary cores")
    valid = bool(result and result.solved and len(result.assignment) == len(evaluator.colors)
        and all(result.assignment[c] in values for c, values in domains.items())
        and all(core[{c: result.assignment[c] for c in core.colors}] != 0
                for core in evaluator.network.coresDict.values()))
    if result and result.solved and not valid:
        raise AssertionError("Fixed-library oracle returned an invalid full assignment")
    return dict(status=status, solved=bool(result and result.solved), valid=valid,
        unsatisfiable=bool(result and result.unsatisfiable), nodes=solver.nodes,
        seconds=perf_counter() - started, **counters,
        contractions=counters["oracle_contractions"] + counters["application_contractions"])


def _worker_init(library, limits):
    global _worker_evaluator
    _worker_evaluator = LibraryEvaluator(library, **limits)


def _worker_score(record, sizes, reference_candidates, reference_steps):
    return _worker_evaluator.score_state(record, sizes=sizes,
        reference_candidates=reference_candidates, reference_steps=reference_steps)


def evaluate(library_path, states_path, output, *, sizes=(), workers=1,
             max_work=1_000_000, max_size=100_000, reference_candidates=0,
             reference_steps=None, resume=False):
    """Persist each completed state atomically; resume only an identical protocol.

    Use one process running this function per output directory. Workers share no
    solver states; each owns one reusable evaluator and its live-mask plans.
    """
    if (any(type(v) is not int or v < 1 for v in (workers, max_work, max_size))
            or type(reference_candidates) is not int or reference_candidates < 0
            or reference_steps is not None and (type(reference_steps) is not int or reference_steps < 1)):
        raise ValueError("Expected positive worker/cost/step limits and nonnegative reference count")
    inputs = [Path(library_path).read_bytes(), Path(states_path).read_bytes()]
    library, payload = (json.loads(value) for value in inputs)
    records = payload if isinstance(payload, list) else payload["states"]
    sizes = sorted(set(sizes or (len(library["clusters"]),)))
    if (not records or any(type(row.get("id")) is not int or row["id"] < 0 for row in records)
            or len({row["id"] for row in records}) != len(records)
            or any(type(size) is not int or not 0 <= size <= len(library["clusters"]) for size in sizes)):
        raise ValueError("Expected uniquely identified states and valid library prefixes")
    network_hash = sha256(json.dumps(library["network"], separators=(",", ":")).encode()).hexdigest()
    if isinstance(payload, dict) and (
            "network_sha256" in payload and payload["network_sha256"] != network_hash
            or "colors" in payload and payload["colors"] != library["network"]["colors"]
            or "dimensions" in payload and payload["dimensions"] != library["network"]["dimensions"]):
        raise ValueError("Snapshot domains and library must use the identical network schema")
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__).with_name(name) for name in ("library_evaluation.py", "library.py",
        "runtime.py", "candidate_evaluation_mac.py", "candidate_evaluation_tnreason.py",
        "candidate_generation.py", "contraction_cost.py", "data.py")]
    paths += [root / "experiments" / "constraint_networks" / name for name in
              ("mac.py", "constraint_networks.py", "generalized_arc_consistency.py")]
    code = {str(path.relative_to(root)): path.read_bytes() for path in paths}
    limits = dict(max_work=max_work, max_size=max_size)
    config = dict(sizes=sizes, workers=workers, **limits, reference_candidates=reference_candidates,
                  reference_steps=reference_steps, reference_cyclic_only=True)
    protocol = dict(version=1, kind="fixed_library_coverage", config=config,
        library_sha256=sha256(inputs[0]).hexdigest(), states_sha256=sha256(inputs[1]).hexdigest(),
        network_sha256=network_hash, code_sha256={key: sha256(value).hexdigest() for key, value in code.items()})
    output = Path(output)
    if resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume requires identical inputs, code and budgets")
        if (output / "summary.json").exists():
            return json.loads((output / "summary.json").read_text())
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "rows").mkdir()
        _write(output / "protocol.json", protocol)
        _write(output / "runtime.json", dict(started_utc=datetime.now(timezone.utc).isoformat(),
            python=platform.python_version(), numpy=np.__version__,
            library=str(Path(library_path).resolve()), states=str(Path(states_path).resolve())))
        for relative, value in code.items():
            target = output / "code" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
    pending = [row for row in records if not (output / "rows" / f"{row['id']:06d}.json").exists()]
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(library, limits)) as pool:
        futures = {pool.submit(_worker_score, row, sizes, reference_candidates, reference_steps): row["id"]
                   for row in pending}
        try:
            for number, future in enumerate(as_completed(futures), len(records) - len(pending) + 1):
                result = future.result()
                _write(output / "rows" / f"{futures[future]:06d}.json", result)
                print(f"state {number}/{len(records)}: {result['id']}, "
                      f"{len(result['successes'])} useful clusters, {result['seconds']:.2f}s", flush=True)
        finally:
            for future in futures:
                future.cancel()
    rows = [json.loads((output / "rows" / f"{row['id']:06d}.json").read_text()) for row in records]
    summary = dict(states=len(rows), clusters=len(library["clusters"]),
        oracle_contractions=sum(row["oracle_contractions"] for row in rows),
        prefixes=[dict(size=size, states_useful=sum(row["prefixes"][i]["useful"] for row in rows),
                       mean_best_gain=sum(row["prefixes"][i]["best_gain"] for row in rows) / len(rows),
                       mean_best_primary_gain=sum(row["prefixes"][i]["best_primary_gain"] for row in rows) / len(rows))
                  for i, size in enumerate(sizes)])
    _write(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library")
    parser.add_argument("states")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=())
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-work", type=int, default=1_000_000)
    parser.add_argument("--max-size", type=int, default=100_000)
    parser.add_argument("--reference-candidates", type=int, default=0)
    parser.add_argument("--reference-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    options = vars(parser.parse_args())
    library_path, states_path = options.pop("library"), options.pop("states")
    print(json.dumps(evaluate(library_path, states_path, **options), indent=2), flush=True)
