from dataclasses import dataclass, field
from typing import Dict, Tuple

from .actions import AlphaReasonAction


@dataclass(frozen=True)
class ClusterCandidate:
    key: str
    colors: Tuple[str, ...]


@dataclass
class AlphaReasonState:
    task_name: str
    evidence: Dict[str, int]
    target_feature_keys: Tuple[str, ...]
    feature_domain_sizes: Dict[str, int]
    target_assignments: Dict[str, int] = field(default_factory=dict)
    feature_priors: Dict[str, Tuple[float, ...]] = field(default_factory=dict)
    legal_actions: Tuple[AlphaReasonAction, ...] = field(default_factory=tuple)
    action_priors: Dict[AlphaReasonAction, float] = field(default_factory=dict)
    active_inference_clusters: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    cluster_candidates: Tuple[ClusterCandidate, ...] = field(default_factory=tuple)
    contradiction: bool = False
    consistency_ok: bool = False
    solved: bool = False
    value_estimate: float = 0.0
    assigned_count: int = 0
    message_count: int = 0

    @property
    def done(self) -> bool:
        return self.contradiction or self.solved

    def state_key(self) -> Tuple[Tuple[Tuple[str, int], ...], Tuple[Tuple[str, Tuple[str, ...]], ...]]:
        return (
            tuple(sorted(self.evidence.items())),
            tuple(sorted(self.active_inference_clusters.items())),
        )

