from .actions import AddClusterAction, AssignAction, ClusterCandidate
from .policies import (
    AlphaSudokuPolicyValueNet,
    HeuristicPolicyValue,
    PolicyValueModelOutput,
    TrainablePolicyValue,
    action_to_key,
    key_to_action,
)
from .state import AlphaSudokuState


class DummyModel:
    def predict(self, state, legal_actions):
        action_scores = {
            action_to_key(AssignAction("a_0_0_0_0_1")): 9.0,
            action_to_key(AddClusterAction("cluster_a")): 1.0,
        }
        return PolicyValueModelOutput(action_scores=action_scores, value=0.75)


def _sample_state():
    return AlphaSudokuState(
        evidence={},
        active_inference_clusters={},
        assignment_actions=("a_0_0_0_0_0", "a_0_0_0_0_1"),
        cluster_candidates=(ClusterCandidate(key="cluster_a", colors=("x", "y")),),
        atom_priors={
            "a_0_0_0_0_0": 0.8,
            "a_0_0_0_0_1": 0.2,
        },
        value_estimate=0.2,
    )


def test_action_key_roundtrip():
    assign_action = AssignAction("a_0_0_0_0_1")
    cluster_action = AddClusterAction("cluster_a")

    assert key_to_action(action_to_key(assign_action)) == assign_action
    assert key_to_action(action_to_key(cluster_action)) == cluster_action


def test_trainable_policy_uses_model_scores():
    state = _sample_state()
    legal_actions = [
        AssignAction("a_0_0_0_0_0"),
        AssignAction("a_0_0_0_0_1"),
        AddClusterAction("cluster_a"),
    ]

    policy = TrainablePolicyValue(model=DummyModel())
    priors = policy.action_priors(state, legal_actions)

    assert priors[AssignAction("a_0_0_0_0_1")] > priors[AssignAction("a_0_0_0_0_0")]
    assert priors[AssignAction("a_0_0_0_0_1")] > priors[AddClusterAction("cluster_a")]
    assert abs(sum(priors.values()) - 1.0) < 1e-9
    assert policy.value(state) == 0.75


def test_trainable_policy_can_blend_with_heuristic():
    state = _sample_state()
    legal_actions = [
        AssignAction("a_0_0_0_0_0"),
        AssignAction("a_0_0_0_0_1"),
        AddClusterAction("cluster_a"),
    ]

    heuristic = HeuristicPolicyValue()
    heuristic_priors = heuristic.action_priors(state, legal_actions)

    policy = TrainablePolicyValue(
        model=DummyModel(),
        fallback_policy=heuristic,
        policy_mix=0.5,
        value_mix=0.5,
    )
    blended_priors = policy.action_priors(state, legal_actions)

    assert abs(sum(blended_priors.values()) - 1.0) < 1e-9
    assert blended_priors != heuristic_priors
    assert policy.value(state) == 0.475


def test_shared_policy_value_net_predicts_scores_and_value():
    state = _sample_state()
    legal_actions = [
        AssignAction("a_0_0_0_0_0"),
        AssignAction("a_0_0_0_0_1"),
        AddClusterAction("cluster_a"),
    ]

    model = AlphaSudokuPolicyValueNet(digit_count=9, hidden_dim=64, action_hidden_dim=32)
    output = model.predict(state, legal_actions)

    assert output.action_scores is not None
    assert set(output.action_scores.keys()) == {action_to_key(action) for action in legal_actions}
    assert output.value is not None
    assert -1.0 <= output.value <= 1.0


def test_trainable_policy_accepts_shared_policy_value_net():
    state = _sample_state()
    legal_actions = [
        AssignAction("a_0_0_0_0_0"),
        AssignAction("a_0_0_0_0_1"),
        AddClusterAction("cluster_a"),
    ]

    model = AlphaSudokuPolicyValueNet(digit_count=9, hidden_dim=64, action_hidden_dim=32)
    policy = TrainablePolicyValue(model=model, policy_mix=1.0, value_mix=1.0)
    priors = policy.action_priors(state, legal_actions)

    assert set(priors.keys()) == set(legal_actions)
    assert abs(sum(priors.values()) - 1.0) < 1e-9
    assert -1.0 <= policy.value(state) <= 1.0
