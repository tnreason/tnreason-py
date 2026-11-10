"""Independent, domain-conditioned cluster trials on an already-closed MAC state."""
from dataclasses import dataclass
from math import prod
from time import perf_counter
from types import SimpleNamespace
import numpy as np
from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.constraint_networks.mac import PropagationResult
from .candidate_evaluation_tnreason import SummaryClusterEvaluation
from .contraction_cost import ContractionEstimate, estimate_candidate_cost


@dataclass(frozen=True)
class CandidateEvaluation(SummaryClusterEvaluation):
    estimate: ContractionEstimate
    propagation: PropagationResult


def prepare_candidate(state, candidate, *, max_work=1_000_000, max_size=100_000,
                      compact_domains=False):
    """Plan without contracting; optionally slice axes to current domain values.

    Dense plans reference live masks and remain valid across domain updates.
    Compact plans contain domain snapshots and must be rebuilt after changes.
    Their optional third item maps the result back to the original axis sizes.
    """
    cores = {key: state._cores[key] for key in candidate.core_keys}
    colors = tuple(dict.fromkeys(color for core in cores.values() for color in core.colors))
    if len(colors) > 52:
        raise ValueError("NumPy contraction supports at most 52 distinct colors")
    indices = {color: tuple(state.domains[color]) for color in colors} if compact_domains else None
    domains = {} if compact_domains else {
        ("domain", color): SimpleNamespace(colors=(color,), shape=(state.dimensions[color],))
        for color in colors}
    inputs = ({key: SimpleNamespace(colors=core.colors, shape=tuple(len(indices[color]) for color in core.colors))
               for key, core in cores.items()} if compact_domains else {**cores, **domains})
    plan = SimpleNamespace(core_keys=(*candidate.core_keys, *domains), open_colors=candidate.open_colors)
    estimate = estimate_candidate_cost(inputs, plan)
    if compact_domains:
        shape = tuple(state.dimensions[color] for color in candidate.open_colors)
        estimate = ContractionEstimate(estimate.work, max(estimate.max_size, prod(shape)),
                                       prod(shape), estimate.path)
    if estimate.work > max_work or estimate.max_size > max_size:
        raise ValueError("Candidate exceeds the contraction work or tensor-size budget")
    idx = {color: axis for axis, color in enumerate(colors)}
    arguments = []
    for key in candidate.core_keys:
        core = cores[key]
        if key in state._contractions:
            support = state._contractions[key][0]
        else:
            support = np.array([core[dict(zip(core.colors, values))] != 0
                                for values in np.ndindex(tuple(core.shape))], dtype=bool).reshape(core.shape)
        if compact_domains:
            support = support[np.ix_(*(indices[color] for color in core.colors))]
        arguments.extend((support, [idx[color] for color in core.colors]))
    if not compact_domains:
        for color in colors:
            arguments.extend((state._domain_tensors[color], [idx[color]]))
    arguments.append([idx[color] for color in candidate.open_colors])
    if compact_domains:
        return estimate, arguments, (shape, tuple(indices[color] for color in candidate.open_colors))
    return estimate, arguments


def contract_candidate(candidate, prepared):
    estimate, arguments = prepared[:2]
    values = np.einsum(*arguments, optimize=["einsum_path", *estimate.path])
    if len(prepared) == 3:
        shape, indices = prepared[2]
        expanded = np.zeros(shape, dtype=bool)
        expanded[np.ix_(*indices)] = values
        values = expanded
    return NumpyCore(values=values, colors=list(candidate.open_colors))


def evaluate_candidate(state, candidate, *, cost_weight=1.0, contradiction_reward=0.0,
                       reward_colors=None, max_work=1_000_000, max_size=100_000,
                       compact_domains=False):
    """Evaluate from GAC closure, then restore domains, cores, hooks and counters.

    Domain-conditioned Boolean contraction uses the planned path. Cost includes
    preparation, planning, contraction, GAC and rollback in seconds. Reward is
    removed values - cost_weight * cost, plus an optional contradiction bonus.
    All colors count unless reward_colors is supplied (e.g. Sudoku's pos_*).
    """
    if state.contradiction or state._queue or state._queued:
        raise ValueError("Evaluate a noncontradictory state after GAC closure")
    started = perf_counter()
    before = state.snapshot_domains()
    targets = tuple(dict.fromkeys(before if reward_colors is None else reward_colors))
    if set(targets) - before.keys():
        raise KeyError("Unknown reward colors")
    summary_key = f"summary_{candidate.key}"
    if summary_key in state._cores:
        raise ValueError("Candidate summary key is already active")
    prepared = prepare_candidate(state, candidate, max_work=max_work, max_size=max_size,
                                 compact_domains=compact_domains)
    estimate = prepared[0]
    summary = contract_candidate(candidate, prepared)

    counters, hook = vars(state.statistics), state.on_closure
    previously_added = summary_key in state._added_core_keys
    state.push_level()
    state.on_closure = None
    try:
        state.add_cores({summary_key: summary}, reversible=True)
        propagation = state.propagate()
        after = state.snapshot_domains()
    finally:
        state.pop_level()
        state.on_closure = hook
        if not previously_added:
            state._added_core_keys.discard(summary_key)
        for name, value in counters.items():
            setattr(state, "_" + name, value)

    removed = sum(len(before[color]) - len(after[color]) for color in targets)
    cost = perf_counter() - started
    reward = removed - cost_weight * cost + contradiction_reward * propagation.contradiction
    return CandidateEvaluation(candidate, after, propagation.contradiction, removed, cost, reward,
                                summary_key, estimate, propagation)
