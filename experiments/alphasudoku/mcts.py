from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, List, Optional, Tuple

from .actions import AlphaSudokuAction
from .env import AlphaSudokuEnv
from .policies import HeuristicPolicyValue, PolicyValue
from .state import AlphaSudokuState


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
    state: AlphaSudokuState
    expanded: bool = False
    children: Dict[AlphaSudokuAction, "SearchNode"] = field(default_factory=dict)
    stats: Dict[AlphaSudokuAction, EdgeStats] = field(default_factory=dict)


class AlphaSudokuMCTS:
    def __init__(
        self,
        env: AlphaSudokuEnv,
        policy_value: Optional[PolicyValue] = None,
        c_puct: float = 1.5,
        simulations: int = 200,
    ):
        self.env = env
        self.policy_value = policy_value or HeuristicPolicyValue()
        self.c_puct = c_puct
        self.simulations = simulations

    def choose_action(self, root_state: AlphaSudokuState) -> AlphaSudokuAction:
        root = SearchNode(state=root_state)
        self._expand(root)

        for _ in range(self.simulations):
            self._simulate(root)

        if not root.stats:
            raise ValueError("No legal actions available at root.")
        return max(root.stats.items(), key=lambda item: item[1].visits)[0]

    def _simulate(self, root: SearchNode) -> None:
        node = root
        path: List[Tuple[SearchNode, AlphaSudokuAction]] = []
        rollout_value = 0.0

        while True:
            if node.state.done:
                rollout_value = node.state.value_estimate
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
                if child_state.done:
                    rollout_value = reward
                else:
                    rollout_value = reward + self._expand(child)
                node = child
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

    def _select(self, node: SearchNode) -> AlphaSudokuAction:
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
