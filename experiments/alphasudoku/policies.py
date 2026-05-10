from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

import torch
from torch import nn

from .actions import AddClusterAction, AlphaSudokuAction, AssignAction
from .state import AlphaSudokuState


def action_to_key(action: AlphaSudokuAction) -> str:
    if isinstance(action, AssignAction):
        return f"assign:{action.atom_key}"
    if isinstance(action, AddClusterAction):
        return f"cluster:{action.cluster_key}"
    raise TypeError(f"Unsupported action type: {type(action)!r}")


def key_to_action(action_key: str) -> AlphaSudokuAction:
    if action_key.startswith("assign:"):
        return AssignAction(action_key.split(":", 1)[1])
    if action_key.startswith("cluster:"):
        return AddClusterAction(action_key.split(":", 1)[1])
    raise ValueError(f"Unsupported action key: {action_key}")


def _parse_atom_key(atom_key: str):
    parts = atom_key.split("_")
    if len(parts) != 6 or parts[0] != "a":
        return None
    r1, r2, c1, c2, digit = (int(part) for part in parts[1:])
    return r1, r2, c1, c2, digit


def _infer_digit_count(state: AlphaSudokuState) -> int:
    max_digit = -1
    atom_keys = list(state.atom_priors.keys()) + list(state.assignment_actions)
    atom_keys.extend(key for key in state.evidence if key.startswith("a_"))
    for atom_key in atom_keys:
        parsed = _parse_atom_key(atom_key)
        if parsed is None:
            continue
        max_digit = max(max_digit, parsed[4])
    return max(1, max_digit + 1) if max_digit >= 0 else 9


def _atom_index(atom_key: str, digit_count: int) -> Optional[int]:
    parsed = _parse_atom_key(atom_key)
    if parsed is None:
        return None
    r1, r2, c1, c2, digit = parsed
    if digit >= digit_count:
        return None
    row = r1 * digit_count + r2
    col = c1 * digit_count + c2
    if row >= digit_count * digit_count or col >= digit_count * digit_count:
        return None
    cell_index = row * (digit_count * digit_count) + col
    return cell_index * digit_count + digit


@dataclass
class PolicyValueModelOutput:
    action_scores: Optional[Dict[str, float]] = None
    value: Optional[float] = None


class PolicyValue(Protocol):
    def action_priors(
        self,
        state: AlphaSudokuState,
        legal_actions: List[AlphaSudokuAction],
    ) -> Dict[AlphaSudokuAction, float]:
        ...

    def value(self, state: AlphaSudokuState) -> float:
        ...


class TrainablePolicyModel(Protocol):
    def predict(
        self,
        state: AlphaSudokuState,
        legal_actions: List[AlphaSudokuAction],
    ) -> PolicyValueModelOutput:
        ...


class AlphaSudokuPolicyValueNet(nn.Module):
    """
    Shared policy/value network with a simple fixed-size Sudoku encoding.

    The trunk encodes the closed AlphaSudoku state. Legal actions are encoded
    separately and scored by an action head conditioned on the shared trunk.
    """

    def __init__(
        self,
        digit_count: int = 9,
        hidden_dim: int = 256,
        action_hidden_dim: int = 128,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        self.digit_count = digit_count
        self.atom_space = digit_count ** 5
        self.state_dim = 2 * self.atom_space + 4
        self.action_dim = self.atom_space + 4
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

    def encode_state(self, state: AlphaSudokuState) -> torch.Tensor:
        assigned = torch.zeros(self.atom_space, dtype=torch.float32, device=self.device)
        priors = torch.zeros(self.atom_space, dtype=torch.float32, device=self.device)

        for atom_key, value in state.evidence.items():
            if value != 1 or not atom_key.startswith("a_"):
                continue
            idx = _atom_index(atom_key, self.digit_count)
            if idx is not None:
                assigned[idx] = 1.0

        for atom_key, prior in state.atom_priors.items():
            idx = _atom_index(atom_key, self.digit_count)
            if idx is not None:
                priors[idx] = float(prior)

        summary = torch.tensor(
            [
                float(state.assigned_count) / max(1.0, float(self.digit_count ** 4)),
                float(len(state.active_inference_clusters)) / 32.0,
                float(len(state.cluster_candidates)) / 32.0,
                float(state.value_estimate),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return torch.cat([assigned, priors, summary], dim=0)

    def encode_actions(self, legal_actions: List[AlphaSudokuAction]) -> torch.Tensor:
        if not legal_actions:
            return torch.zeros((0, self.action_dim), dtype=torch.float32, device=self.device)

        action_vectors = []
        for action in legal_actions:
            vector = torch.zeros(self.action_dim, dtype=torch.float32, device=self.device)
            if isinstance(action, AssignAction):
                idx = _atom_index(action.atom_key, self.digit_count)
                if idx is not None:
                    vector[idx] = 1.0
                vector[self.atom_space] = 1.0
            elif isinstance(action, AddClusterAction):
                vector[self.atom_space + 1] = 1.0
                cluster_size = len(action.cluster_key.split("_"))
                vector[self.atom_space + 2] = min(1.0, cluster_size / 16.0)
                vector[self.atom_space + 3] = min(1.0, len(action.cluster_key) / 128.0)
            action_vectors.append(vector)
        return torch.stack(action_vectors, dim=0)

    def forward_tensors(
        self,
        state_tensor: torch.Tensor,
        action_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_hidden = self.trunk(state_tensor.unsqueeze(0)).squeeze(0)
        value = self.value_head(state_hidden).squeeze(-1)
        if action_tensor.shape[0] == 0:
            return torch.zeros(0, dtype=torch.float32, device=self.device), value

        expanded_hidden = state_hidden.unsqueeze(0).expand(action_tensor.shape[0], -1)
        action_scores = self.action_head(torch.cat([expanded_hidden, action_tensor], dim=1)).squeeze(-1)
        return action_scores, value

    def predict(
        self,
        state: AlphaSudokuState,
        legal_actions: List[AlphaSudokuAction],
    ) -> PolicyValueModelOutput:
        with torch.no_grad():
            state_tensor = self.encode_state(state)
            action_tensor = self.encode_actions(legal_actions)
            action_scores, value = self.forward_tensors(state_tensor, action_tensor)
            scores = {
                action_to_key(action): float(score)
                for action, score in zip(legal_actions, action_scores.tolist())
            }
            return PolicyValueModelOutput(action_scores=scores, value=float(value.item()))


class HeuristicPolicyValue:
    def action_priors(
        self,
        state: AlphaSudokuState,
        legal_actions: List[AlphaSudokuAction],
    ) -> Dict[AlphaSudokuAction, float]:
        if not legal_actions:
            return {}

        assign_actions = [action for action in legal_actions if isinstance(action, AssignAction)]
        cluster_actions = [action for action in legal_actions if isinstance(action, AddClusterAction)]

        priors: Dict[AlphaSudokuAction, float] = {}
        if assign_actions:
            assign_weights = {
                action: max(state.atom_priors.get(action.atom_key, 0.0), 1e-6)
                for action in assign_actions
            }
            assign_total = sum(assign_weights.values())
            assign_mass = 0.7 if cluster_actions else 1.0
            for action, weight in assign_weights.items():
                priors[action] = assign_mass * weight / assign_total

        if cluster_actions:
            cluster_weights = {
                action: 1.0 / max(
                    1,
                    len(next(candidate.colors for candidate in state.cluster_candidates if candidate.key == action.cluster_key)),
                )
                for action in cluster_actions
            }
            cluster_total = sum(cluster_weights.values())
            cluster_mass = 0.3 if assign_actions else 1.0
            for action, weight in cluster_weights.items():
                priors[action] = cluster_mass * weight / cluster_total

        return _normalize_action_distribution(priors, legal_actions)

    def value(self, state: AlphaSudokuState) -> float:
        return state.value_estimate


class TrainablePolicyValue:
    """
    Adapter for learned policy/value models.

    The model only needs to expose:

    `predict(state, legal_actions) -> PolicyValueModelOutput`

    where `action_scores` is a mapping from stable action keys like
    `assign:a_0_0_0_0_1` or `cluster:some_cluster_key` to non-negative weights
    or logits, and `value` is a scalar in roughly [-1, 1].

    The class can blend model outputs with a heuristic fallback, which is useful
    early in training when the learned policy is still weak.
    """

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
        state: AlphaSudokuState,
        legal_actions: List[AlphaSudokuAction],
    ) -> Dict[AlphaSudokuAction, float]:
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

    def value(self, state: AlphaSudokuState) -> float:
        fallback_value = self.fallback_policy.value(state)
        model_output = self.model.predict(state, [])
        if model_output.value is None:
            return fallback_value
        model_value = max(-1.0, min(1.0, float(model_output.value)))
        return self.value_mix * model_value + (1.0 - self.value_mix) * fallback_value

    def _action_priors_from_output(
        self,
        model_output: PolicyValueModelOutput,
        legal_actions: List[AlphaSudokuAction],
    ) -> Dict[AlphaSudokuAction, float]:
        if not model_output.action_scores:
            return {}

        scored_actions: Dict[AlphaSudokuAction, float] = {}
        legal_action_keys = {action_to_key(action): action for action in legal_actions}
        for action_key, score in model_output.action_scores.items():
            if action_key not in legal_action_keys:
                continue
            scored_actions[legal_action_keys[action_key]] = float(score)
        return _softmax_action_distribution(scored_actions, legal_actions)


def _normalize_action_distribution(
    priors: Dict[AlphaSudokuAction, float],
    legal_actions: List[AlphaSudokuAction],
) -> Dict[AlphaSudokuAction, float]:
    filtered = {
        action: max(0.0, float(priors.get(action, 0.0)))
        for action in legal_actions
    }
    total = sum(filtered.values())
    if total <= 0.0:
        uniform = 1.0 / len(legal_actions)
        return {action: uniform for action in legal_actions}
    return {action: value / total for action, value in filtered.items()}


def _softmax_action_distribution(
    scores: Dict[AlphaSudokuAction, float],
    legal_actions: List[AlphaSudokuAction],
) -> Dict[AlphaSudokuAction, float]:
    if not legal_actions:
        return {}
    filtered_scores = {
        action: float(scores.get(action, float("-inf")))
        for action in legal_actions
    }
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
