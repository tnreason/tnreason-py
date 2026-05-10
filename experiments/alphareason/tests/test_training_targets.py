from ..actions import AssignValueAction
from ..state import AlphaReasonState
from ..task import AlphaReasonTask
from ..training_targets import policy_target_from_solution, value_target_from_solution


def test_policy_target_is_uniform_over_solution_consistent_actions():
    state = AlphaReasonState(
        task_name="toy",
        evidence={},
        target_feature_keys=("x", "y"),
        feature_domain_sizes={"x": 2, "y": 2},
        legal_actions=(
            AssignValueAction("x", 0),
            AssignValueAction("x", 1),
            AssignValueAction("y", 1),
        ),
    )
    task = AlphaReasonTask(
        name="toy",
        network_factory=lambda evidence: None,
        initial_evidence={},
        target_feature_keys=("x", "y"),
        feature_domain_sizes={"x": 2, "y": 2},
        assignment_evidence_map={"x": {0: {"x": 0}, 1: {"x": 1}}, "y": {0: {"y": 0}, 1: {"y": 1}}},
        solution_assignments={"x": 1, "y": 1},
    )

    target = policy_target_from_solution(state, task)
    assert target[AssignValueAction("x", 1)] == 0.5
    assert target[AssignValueAction("y", 1)] == 0.5
    assert AssignValueAction("x", 0) not in target


def test_value_target_is_negative_for_off_solution_assignment():
    state = AlphaReasonState(
        task_name="toy",
        evidence={},
        target_feature_keys=("x",),
        feature_domain_sizes={"x": 2},
        target_assignments={"x": 0},
    )
    task = AlphaReasonTask(
        name="toy",
        network_factory=lambda evidence: None,
        initial_evidence={},
        target_feature_keys=("x",),
        feature_domain_sizes={"x": 2},
        assignment_evidence_map={"x": {0: {"x": 0}, 1: {"x": 1}}},
        solution_assignments={"x": 1},
    )

    assert value_target_from_solution(state, task) == -1.0

