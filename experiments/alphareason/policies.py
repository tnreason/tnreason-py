from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

import torch
from torch import nn

from .actions import AddClusterAction, AlphaReasonAction, AssignValueAction
from .state import AlphaReasonState


def action_to_key(action: AlphaReasonAction) -> str:
    if isinstance(action, AssignValueAction):
        return f"assign:{action.feature_key}:{action.value_index}"
    if isinstance(action, AddClusterAction):
        return f"cluster:{action.cluster_key}"
    raise TypeError(f"Unsupported action type: {type(action)!r}")


@dataclass
class PolicyValueModelOutput:
    action_scores: Optional[Dict[str, float]] = None
    value: Optional[float] = None


class PolicyValue(Protocol):
    def action_priors(
        self,
        state: AlphaReasonState,
        legal_actions: List[AlphaReasonAction],
    ) -> Dict[AlphaReasonAction, float]:
        ...

    def value(self, state: AlphaReasonState) -> float:
        ...


class TrainablePolicyModel(Protocol):
    def predict(
        self,
        state: AlphaReasonState,
        legal_actions: List[AlphaReasonAction],
    ) -> PolicyValueModelOutput:
        ...


class AlphaReasonPolicyValueNet(nn.Module):
    def __init__(
        self,
        num_features: int,
        max_feature_dim: int,
        hidden_dim: int = 256,
        action_hidden_dim: int = 128,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.max_feature_dim = max_feature_dim
        self.state_dim = num_features * max_feature_dim * 2 + num_features * 2 + 4
        self.action_dim = num_features + max_feature_dim + 3
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self.trunk = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim + self.action_dim, action_hidden_dim),
            nn.ReLU(),
            nn.Linear(action_hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )
        self.to(self.device)

    def encode_state(self, state: AlphaReasonState) -> torch.Tensor:
        assigned = torch.zeros((self.num_features, self.max_feature_dim), dtype=torch.float32, device=self.device)
        priors = torch.zeros((self.num_features, self.max_feature_dim), dtype=torch.float32, device=self.device)
        assigned_mask = torch.zeros(self.num_features, dtype=torch.float32, device=self.device)
        unresolved_mask = torch.zeros(self.num_features, dtype=torch.float32, device=self.device)

        feature_index = {feature_key: idx for idx, feature_key in enumerate(state.target_feature_keys[: self.num_features])}
        for feature_key, idx in feature_index.items():
            domain_size = min(state.feature_domain_sizes[feature_key], self.max_feature_dim)
            feature_probs = state.feature_priors.get(feature_key, tuple())
            for value_index in range(min(len(feature_probs), domain_size)):
                priors[idx, value_index] = float(feature_probs[value_index])
            if feature_key in state.target_assignments:
                value_index = state.target_assignments[feature_key]
                if value_index < self.max_feature_dim:
                    assigned[idx, value_index] = 1.0
                assigned_mask[idx] = 1.0
            else:
                unresolved_mask[idx] = 1.0

        summary = torch.tensor(
            [
                float(state.assigned_count) / max(1.0, float(len(state.target_feature_keys))),
                float(len(state.active_inference_clusters)) / 32.0,
                float(len(state.cluster_candidates)) / 32.0,
                float(state.value_estimate),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return torch.cat(
            [
                assigned.flatten(),
                priors.flatten(),
                assigned_mask,
                unresolved_mask,
                summary,
            ],
            dim=0,
        )

    def encode_actions(self, state: AlphaReasonState, legal_actions: List[AlphaReasonAction]) -> torch.Tensor:
        if not legal_actions:
            return torch.zeros((0, self.action_dim), dtype=torch.float32, device=self.device)

        feature_index = {feature_key: idx for idx, feature_key in enumerate(state.target_feature_keys[: self.num_features])}
        action_vectors = []
        for action in legal_actions:
            vector = torch.zeros(self.action_dim, dtype=torch.float32, device=self.device)
            if isinstance(action, AssignValueAction):
                if action.feature_key in feature_index:
                    vector[feature_index[action.feature_key]] = 1.0
                if action.value_index < self.max_feature_dim:
                    vector[self.num_features + action.value_index] = 1.0
                vector[self.num_features + self.max_feature_dim] = 1.0
            elif isinstance(action, AddClusterAction):
                vector[self.num_features + self.max_feature_dim + 1] = 1.0
                vector[self.num_features + self.max_feature_dim + 2] = min(1.0, len(action.cluster_key) / 128.0)
            action_vectors.append(vector)
        return torch.stack(action_vectors, dim=0)

    def forward_tensors(self, state_tensor: torch.Tensor, action_tensor: torch.Tensor):
        state_hidden = self.trunk(state_tensor.unsqueeze(0)).squeeze(0)
        value = self.value_head(state_hidden).squeeze(-1)
        if action_tensor.shape[0] == 0:
            return torch.zeros(0, dtype=torch.float32, device=self.device), value

        expanded_hidden = state_hidden.unsqueeze(0).expand(action_tensor.shape[0], -1)
        action_scores = self.action_head(torch.cat([expanded_hidden, action_tensor], dim=1)).squeeze(-1)
        return action_scores, value

    def predict(
        self,
        state: AlphaReasonState,
        legal_actions: List[AlphaReasonAction],
    ) -> PolicyValueModelOutput:
        with torch.no_grad():
            state_tensor = self.encode_state(state)
            action_tensor = self.encode_actions(state, legal_actions)
            action_scores, value = self.forward_tensors(state_tensor, action_tensor)
            return PolicyValueModelOutput(
                action_scores={action_to_key(action): float(score) for action, score in zip(legal_actions, action_scores.tolist())},
                value=float(value.item()),
            )


class HeuristicPolicyValue:
    def action_priors(
        self,
        state: AlphaReasonState,
        legal_actions: List[AlphaReasonAction],
    ) -> Dict[AlphaReasonAction, float]:
        priors = {
            action: state.action_priors.get(action, 0.0)
            for action in legal_actions
        }
        return _normalize_action_distribution(priors, legal_actions)

    def value(self, state: AlphaReasonState) -> float:
        return state.value_estimate


class TrainablePolicyValue:
    def __init__(
        self,
        model: TrainablePolicyModel,
        fallback_policy: Optional[PolicyValue] = None,
        policy_mix: float = 1.0,
        value_mix: float = 1.0,
    ):
        self.model = model
        self.fallback_policy = fallback_policy or HeuristicPolicyValue()
        self.policy_mix = max(0.0, min(1.0, policy_mix))
        self.value_mix = max(0.0, min(1.0, value_mix))

    def action_priors(
        self,
        state: AlphaReasonState,
        legal_actions: List[AlphaReasonAction],
    ) -> Dict[AlphaReasonAction, float]:
        if not legal_actions:
            return {}

        fallback_priors = self.fallback_policy.action_priors(state, legal_actions)
        model_output = self.model.predict(state, legal_actions)
        model_priors = self._action_priors_from_output(model_output, legal_actions)

        if self.policy_mix <= 0.0:
            return fallback_priors
        if self.policy_mix >= 1.0 and model_priors:
            return model_priors

        blended = {}
        for action in legal_actions:
            blended[action] = (
                self.policy_mix * model_priors.get(action, 0.0)
                + (1.0 - self.policy_mix) * fallback_priors.get(action, 0.0)
            )
        return _normalize_action_distribution(blended, legal_actions)

    def value(self, state: AlphaReasonState) -> float:
        fallback_value = self.fallback_policy.value(state)
        model_output = self.model.predict(state, [])
        if model_output.value is None:
            return fallback_value
        model_value = max(-1.0, min(1.0, float(model_output.value)))
        return self.value_mix * model_value + (1.0 - self.value_mix) * fallback_value

    def _action_priors_from_output(
        self,
        model_output: PolicyValueModelOutput,
        legal_actions: List[AlphaReasonAction],
    ) -> Dict[AlphaReasonAction, float]:
        if not model_output.action_scores:
            return {}

        scored_actions = {}
        legal_action_keys = {action_to_key(action): action for action in legal_actions}
        for action_key, score in model_output.action_scores.items():
            if action_key not in legal_action_keys:
                continue
            scored_actions[legal_action_keys[action_key]] = float(score)
        return _softmax_action_distribution(scored_actions, legal_actions)


def _normalize_action_distribution(
    priors: Dict[AlphaReasonAction, float],
    legal_actions: List[AlphaReasonAction],
) -> Dict[AlphaReasonAction, float]:
    if not legal_actions:
        return {}
    filtered = {action: max(0.0, float(priors.get(action, 0.0))) for action in legal_actions}
    total = sum(filtered.values())
    if total <= 0:
        uniform = 1.0 / len(legal_actions)
        return {action: uniform for action in legal_actions}
    return {action: value / total for action, value in filtered.items()}


def _softmax_action_distribution(
    scores: Dict[AlphaReasonAction, float],
    legal_actions: List[AlphaReasonAction],
) -> Dict[AlphaReasonAction, float]:
    if not legal_actions:
        return {}
    filtered_scores = {action: float(scores.get(action, float("-inf"))) for action in legal_actions}
    finite_scores = [score for score in filtered_scores.values() if score != float("-inf")]
    if not finite_scores:
        return _normalize_action_distribution({}, legal_actions)

    max_score = max(finite_scores)
    exp_scores = {}
    for action, score in filtered_scores.items():
        if score == float("-inf"):
            exp_scores[action] = 0.0
        else:
            exp_scores[action] = float(torch.exp(torch.tensor(score - max_score)).item())
    return _normalize_action_distribution(exp_scores, legal_actions)

