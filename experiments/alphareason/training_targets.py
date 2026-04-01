from typing import Dict

from .actions import AssignValueAction
from .state import AlphaReasonState
from .task import AlphaReasonTask


def solution_consistent_actions(state: AlphaReasonState, task: AlphaReasonTask):
    consistent = []
    for action in state.legal_actions:
        if not isinstance(action, AssignValueAction):
            continue
        if task.solution_assignments.get(action.feature_key) == action.value_index:
            consistent.append(action)
    return tuple(consistent)


def policy_target_from_solution(state: AlphaReasonState, task: AlphaReasonTask) -> Dict[object, float]:
    good_actions = solution_consistent_actions(state, task)
    if not good_actions:
        return {}
    weight = 1.0 / len(good_actions)
    return {action: weight for action in good_actions}


def value_target_from_solution(state: AlphaReasonState, task: AlphaReasonTask) -> float:
    if state.contradiction:
        return -1.0
    for feature_key, value_index in state.target_assignments.items():
        if feature_key in task.solution_assignments and task.solution_assignments[feature_key] != value_index:
            return -1.0
    return 1.0

