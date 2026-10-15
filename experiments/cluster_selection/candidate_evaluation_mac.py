"""Independent, domain-conditioned cluster trials on an already-closed MAC state."""
from dataclasses import dataclass
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


def evaluate_candidate(state, candidate, *, cost_weight=1.0, contradiction_reward=0.0,
                           reward_colors=None, max_work=1_000_000, max_size=100_000):
    """Evaluate a temporary summary, then restore the starting MAC state.

    Call state.propagate() first. Domains enter the contraction as persistent
    unary masks; Boolean supports avoid cancellation and count overflow. The
    planner and execution use the same inputs/path and original dense shapes.
    Closure hooks are suspended so only this candidate drives the trial.

    Cost is elapsed seconds for preparation, planning, contraction, GAC and
    rollback. Reward is removed values - cost_weight * cost, plus an optional
    contradiction bonus. All colors count unless reward_colors is supplied
    (e.g. Sudoku's pos_* variables). This does not commit any learned pruning.
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

    cores = {key: state._cores[key] for key in candidate.core_keys}
    colors = tuple(dict.fromkeys(color for core in cores.values() for color in core.colors))
    domains = {("domain", color): SimpleNamespace(colors=(color,), shape=(state.dimensions[color],))
               for color in colors}
    inputs = {**cores, **domains}
    plan = SimpleNamespace(core_keys=(*candidate.core_keys, *domains), open_colors=candidate.open_colors)
    estimate = estimate_candidate_cost(inputs, plan)
    if estimate.work > max_work or estimate.max_size > max_size:
        raise ValueError("Candidate exceeds the contraction work or tensor-size budget")
    if len(colors) > 52:
        raise ValueError("NumPy contraction supports at most 52 distinct colors")
    idx = {color: axis for axis, color in enumerate(colors)}
    arguments = []
    for key in candidate.core_keys:
        core = cores[key]
        if key in state._contractions:
            support = state._contractions[key][0]
        else:
            support = np.array([core[dict(zip(core.colors, values))] != 0
                                for values in np.ndindex(tuple(core.shape))], dtype=bool).reshape(core.shape)
        arguments.extend((support, [idx[color] for color in core.colors]))
    for color in colors:
        arguments.extend((state._domain_tensors[color], [idx[color]]))
    arguments.append([idx[color] for color in candidate.open_colors])
    summary = NumpyCore(values=np.einsum(*arguments, optimize=["einsum_path", *estimate.path]),
                        colors=list(candidate.open_colors))

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
