from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple


def _missing_closure_function(*args, **kwargs):
    raise NotImplementedError("AlphaReasonTask.closure_function must be provided to close this task.")


@dataclass(frozen=True)
class AlphaReasonTask:
    name: str
    network_factory: Callable[[Dict[str, int]], object]
    initial_evidence: Dict[str, int]
    target_feature_keys: Tuple[str, ...]
    feature_domain_sizes: Dict[str, int]
    assignment_evidence_map: Dict[str, Dict[int, Dict[str, int]]]
    closure_function: Callable[[Any, "AlphaReasonTask", Dict[str, int], Dict[str, Tuple[str, ...]]], object] = _missing_closure_function
    action_feature_keys: Tuple[str, ...] = field(default_factory=tuple)
    solution_assignments: Dict[str, int] = field(default_factory=dict)
    rule_detectors: Tuple[Callable[..., List[List[str]]], ...] = field(default_factory=tuple)
    rule_detector_kwargs: Dict[str, object] = field(default_factory=dict)
    consistency_checker: Optional[Callable[[object, Dict[str, int]], bool]] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def assignment_to_evidence(self, feature_key: str, value_index: int) -> Dict[str, int]:
        if feature_key not in self.assignment_evidence_map:
            return {feature_key: value_index}
        if value_index not in self.assignment_evidence_map[feature_key]:
            raise KeyError(f"Unsupported value {value_index} for feature {feature_key}.")
        return dict(self.assignment_evidence_map[feature_key][value_index])

    def solution_action_map(self) -> Mapping[str, int]:
        return self.solution_assignments

    def available_action_feature_keys(self) -> Tuple[str, ...]:
        return self.action_feature_keys or self.target_feature_keys
