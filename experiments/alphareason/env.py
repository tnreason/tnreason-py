from typing import Dict, List, Tuple

from .actions import AddClusterAction, AlphaReasonAction, AssignValueAction
from .closure_engine import AlphaReasonClosureEngine
from .state import AlphaReasonState
from .task import AlphaReasonTask


class AlphaReasonEnv:
    def __init__(self, task: AlphaReasonTask, closure_engine: AlphaReasonClosureEngine):
        self.task = task
        self.closure_engine = closure_engine

    def reset(self) -> AlphaReasonState:
        return self.closure_engine.close(self.task, self.task.initial_evidence, active_inference_clusters={})

    def step(self, state: AlphaReasonState, action: AlphaReasonAction) -> Tuple[AlphaReasonState, float, bool]:
        evidence = dict(state.evidence)
        active_clusters = dict(state.active_inference_clusters)

        if isinstance(action, AssignValueAction):
            evidence.update(self.task.assignment_to_evidence(action.feature_key, action.value_index))
        elif isinstance(action, AddClusterAction):
            cluster_map = {candidate.key: candidate.colors for candidate in state.cluster_candidates}
            if action.cluster_key not in cluster_map:
                raise ValueError(f"Unknown cluster action {action.cluster_key}.")
            active_clusters[action.cluster_key] = cluster_map[action.cluster_key]
        else:
            raise ValueError(f"Unsupported action {action}.")

        next_state = self.closure_engine.close(self.task, evidence, active_inference_clusters=active_clusters)
        reward = self._reward(state, next_state)
        return next_state, reward, next_state.done

    def _reward(self, prev_state: AlphaReasonState, next_state: AlphaReasonState) -> float:
        if next_state.contradiction:
            return -1.0
        if next_state.solved:
            return 1.0
        progress = (next_state.assigned_count - prev_state.assigned_count) / float(max(1, len(self.task.target_feature_keys)))
        return progress

