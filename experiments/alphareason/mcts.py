from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, List, Optional, Tuple

from .actions import AlphaReasonAction
from .env import AlphaReasonEnv
from .policies import HeuristicPolicyValue, PolicyValue
from .state import AlphaReasonState


@dataclass
class EdgeStats:
    prior: float
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


@dataclass
class SearchNode:
    state: AlphaReasonState
    expanded: bool = False
    children: Dict[AlphaReasonAction, "SearchNode"] = field(default_factory=dict)
    stats: Dict[AlphaReasonAction, EdgeStats] = field(default_factory=dict)


class AlphaReasonMCTS:
    def __init__(
        self,
        env: AlphaReasonEnv,
        policy_value: Optional[PolicyValue] = None,
        c_puct: float = 1.5,
        simulations: int = 200,
        rollout_depth: int = 0,
    ):
        self.env = env
        self.policy_value = policy_value or HeuristicPolicyValue()
        self.c_puct = c_puct
        self.simulations = simulations
        self.rollout_depth = rollout_depth

    def choose_action(self, root_state: AlphaReasonState) -> AlphaReasonAction:
        ranked_actions = self.rank_actions(root_state)
        if not ranked_actions:
            raise ValueError("No legal actions available at root.")
        return ranked_actions[0]

    def rank_actions(self, root_state: AlphaReasonState) -> List[AlphaReasonAction]:
        root = SearchNode(state=root_state)
        self._expand(root)

        for _ in range(self.simulations):
            self._simulate(root)

        if not root.stats:
            return []
        return [
            action
            for action, _ in sorted(
                root.stats.items(),
                key=lambda item: (item[1].visits, item[1].q_value, item[1].prior),
                reverse=True,
            )
        ]

    def _simulate(self, root: SearchNode) -> None:
        node = root
        path: List[Tuple[SearchNode, AlphaReasonAction]] = []
        rollout_value = 0.0

        while True:
            if node.state.done:
                rollout_value = self.policy_value.value(node.state)
                break

            if not node.expanded:
                rollout_value = self._expand(node)
                break

            action = self._select(node)
            path.append((node, action))

            if action not in node.children:
                child_state, reward, _ = self.env.step(node.state, action)
                child = SearchNode(state=child_state)
                node.children[action] = child
                rollout_value = reward if child_state.done else reward + self._rollout(child_state)
                break

            node = node.children[action]

        for parent, action in path:
            stats = parent.stats[action]
            stats.visits += 1
            stats.value_sum += rollout_value

    def _expand(self, node: SearchNode) -> float:
        if node.state.done:
            node.expanded = True
            return self.policy_value.value(node.state)

        legal_actions = self.env.legal_actions(node.state)
        priors = self.policy_value.action_priors(node.state, legal_actions)
        node.stats = {action: EdgeStats(prior=priors[action]) for action in legal_actions}
        node.expanded = True
        return self.policy_value.value(node.state)

    def _rollout(self, state: AlphaReasonState) -> float:
        total_value = 0.0
        current_state = state
        for _ in range(self.rollout_depth):
            if current_state.done:
                return total_value + self.policy_value.value(current_state)
            legal_actions = self.env.legal_actions(current_state)
            if not legal_actions:
                return total_value - 1.0
            priors = self.policy_value.action_priors(current_state, legal_actions)
            action = max(legal_actions, key=lambda candidate: priors.get(candidate, 0.0))
            current_state, reward, _ = self.env.step(current_state, action)
            total_value += reward
        return total_value + self.policy_value.value(current_state)

    def _select(self, node: SearchNode) -> AlphaReasonAction:
        total_visits = sum(stats.visits for stats in node.stats.values()) + 1
        sqrt_total = sqrt(total_visits)

        best_action = None
        best_score = float("-inf")
        for action, stats in node.stats.items():
            score = stats.q_value + self.c_puct * stats.prior * sqrt_total / (1 + stats.visits)
            if score > best_score:
                best_score = score
                best_action = action

        if best_action is None:
            raise ValueError("Failed to select an action.")
        return best_action
