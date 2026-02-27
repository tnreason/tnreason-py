from dataclasses import dataclass, field
from typing import Dict, Tuple

from .actions import ClusterCandidate


@dataclass
class AlphaSudokuState:
    evidence: Dict[str, int]
    active_inference_clusters: Dict[str, Tuple[str, ...]]
    assignment_actions: Tuple[str, ...] = field(default_factory=tuple)
    cluster_candidates: Tuple[ClusterCandidate, ...] = field(default_factory=tuple)
    atom_priors: Dict[str, float] = field(default_factory=dict)
    contradiction: bool = False
    consistency_ok: bool = False
    solved: bool = False
    value_estimate: float = 0.0
    assigned_count: int = 0
    message_count: int = 0

    @property
    def done(self) -> bool:
        return self.contradiction or self.solved

    def state_key(self) -> Tuple[Tuple[str, int], Tuple[Tuple[str, Tuple[str, ...]], ...]]:
        return (
            tuple(sorted(self.evidence.items())),
            tuple(sorted(self.active_inference_clusters.items())),
        )

