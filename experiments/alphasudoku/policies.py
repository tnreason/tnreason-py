from typing import Dict, List

from .actions import AddClusterAction, AlphaSudokuAction, AssignAction
from .state import AlphaSudokuState


class HeuristicPolicyValue:
    def action_priors(self, state: AlphaSudokuState, legal_actions: List[AlphaSudokuAction]) -> Dict[AlphaSudokuAction, float]:
        if not legal_actions:
            return {}

        assign_actions = [action for action in legal_actions if isinstance(action, AssignAction)]
        cluster_actions = [action for action in legal_actions if isinstance(action, AddClusterAction)]

        priors: Dict[AlphaSudokuAction, float] = {}
        if assign_actions:
            assign_weights = {action: max(state.atom_priors.get(action.atom_key, 0.0), 1e-6) for action in assign_actions}
            assign_total = sum(assign_weights.values())
            assign_mass = 0.7 if cluster_actions else 1.0
            for action, weight in assign_weights.items():
                priors[action] = assign_mass * weight / assign_total

        if cluster_actions:
            cluster_weights = {
                action: 1.0 / max(1, len(next(candidate.colors for candidate in state.cluster_candidates if candidate.key == action.cluster_key)))
                for action in cluster_actions
            }
            cluster_total = sum(cluster_weights.values())
            cluster_mass = 0.3 if assign_actions else 1.0
            for action, weight in cluster_weights.items():
                priors[action] = cluster_mass * weight / cluster_total

        total = sum(priors.values())
        return {action: value / total for action, value in priors.items()}

    def value(self, state: AlphaSudokuState) -> float:
        return state.value_estimate

