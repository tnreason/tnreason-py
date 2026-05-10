from typing import Dict, List, Tuple

from .actions import AddClusterAction, AlphaSudokuAction, AssignAction
from .closure_engine import AlphaSudokuClosureEngine
from .state import AlphaSudokuState


class AlphaSudokuEnv:
    def __init__(self, closure_engine: AlphaSudokuClosureEngine):
        self.closure_engine = closure_engine

    def reset(self, evidence: Dict[str, int]) -> AlphaSudokuState:
        return self.closure_engine.close(evidence, active_inference_clusters={})

    def legal_actions(self, state: AlphaSudokuState) -> List[AlphaSudokuAction]:
        if state.done:
            return []
        actions: List[AlphaSudokuAction] = [AssignAction(atom_key) for atom_key in state.assignment_actions]
        actions.extend(AddClusterAction(candidate.key) for candidate in state.cluster_candidates)
        return actions

    def step(self, state: AlphaSudokuState, action: AlphaSudokuAction) -> Tuple[AlphaSudokuState, float, bool]:
        evidence = dict(state.evidence)
        active_clusters = dict(state.active_inference_clusters)

        if isinstance(action, AssignAction):
            evidence[action.atom_key] = 1
        elif isinstance(action, AddClusterAction):
            cluster_map = {candidate.key: candidate.colors for candidate in state.cluster_candidates}
            if action.cluster_key not in cluster_map:
                raise ValueError(f"Unknown cluster action {action.cluster_key}.")
            active_clusters[action.cluster_key] = cluster_map[action.cluster_key]
        else:
            raise ValueError(f"Unsupported action {action}.")

        next_state = self.closure_engine.close(evidence, active_inference_clusters=active_clusters)
        reward = self._reward(state, next_state)
        return next_state, reward, next_state.done

    def _reward(self, prev_state: AlphaSudokuState, next_state: AlphaSudokuState) -> float:
        if next_state.contradiction:
            return -1.0
        if next_state.solved:
            return 1.0

        progress = (next_state.assigned_count - prev_state.assigned_count) / float(self.closure_engine.num ** 4)
        return progress

