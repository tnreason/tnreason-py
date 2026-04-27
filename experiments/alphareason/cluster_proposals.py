from dataclasses import dataclass
from typing import Dict, Tuple

from .constraint_propagation import add_cluster_summary_core, enforce_generalized_arc_consistency
from .state import AlphaReasonState, ClusterCandidate
from .actions import cluster_key_from_colors


@dataclass(frozen=True)
class ClusterEvaluation:
    candidate: ClusterCandidate
    removed_supports: int

    @property
    def gain(self) -> float:
        return float(self.removed_supports)


def propose_direct_micro_clusters(
    core_dict,
    state: AlphaReasonState,
    max_candidates: int = 5,
) -> Tuple[ClusterCandidate, ...]:
    target_features = set(state.target_feature_keys)
    active_keys = set(state.active_inference_clusters)
    evaluations = []

    for core in core_dict.values():
        colors = tuple(sorted(color for color in core.colors if color in target_features))
        if len(colors) < 2:
            continue
        key = cluster_key_from_colors(colors)
        if key in active_keys:
            continue
        candidate = ClusterCandidate(
            key=key,
            colors=colors,
            projected_features=colors,
            estimated_cost=float(len(colors)),
            tier="micro",
        )
        evaluation = evaluate_micro_cluster(core_dict, state, candidate)
        if evaluation.removed_supports > 0:
            evaluations.append(evaluation)

    evaluations.sort(key=lambda item: (-item.gain, item.candidate.estimated_cost, item.candidate.key))
    return tuple(
        ClusterCandidate(
            key=item.candidate.key,
            colors=item.candidate.colors,
            projected_features=item.candidate.projected_features,
            estimated_cost=item.candidate.estimated_cost,
            estimated_gain=item.gain,
            tier=item.candidate.tier,
        )
        for item in evaluations[:max_candidates]
    )


def evaluate_micro_cluster(
    core_dict,
    state: AlphaReasonState,
    candidate: ClusterCandidate,
) -> ClusterEvaluation:
    updated = add_cluster_summary_core(core_dict, open_colors=candidate.colors, name=f"candidate_{candidate.key}")
    domains, contradiction = enforce_generalized_arc_consistency(updated)
    if contradiction:
        return ClusterEvaluation(candidate=candidate, removed_supports=sum(len(state.feature_supports.get(color, ())) for color in candidate.colors))

    removed_supports = 0
    for color in candidate.projected_features:
        current_support = set(state.feature_supports.get(color, ()))
        new_support = set(domains.get(color, current_support))
        removed_supports += len(current_support - new_support)
    return ClusterEvaluation(candidate=candidate, removed_supports=removed_supports)
