from ..actions import AddClusterAction, AssignValueAction
from ..policies import (
    AlphaReasonPolicyValueNet,
    HeuristicPolicyValue,
    PolicyValueModelOutput,
    TrainablePolicyValue,
    action_to_key,
)
from ..state import AlphaReasonState, ClusterCandidate


class DummyModel:
    def predict(self, state, legal_actions):
        return PolicyValueModelOutput(
            action_scores={
                action_to_key(AssignValueAction("x", 1)): 5.0,
                action_to_key(AddClusterAction("cluster_a")): 1.0,
            },
            value=0.8,
        )


def _sample_state():
    actions = (
        AssignValueAction("x", 0),
        AssignValueAction("x", 1),
        AddClusterAction("cluster_a"),
    )
    return AlphaReasonState(
        task_name="toy",
        evidence={},
        target_feature_keys=("x", "y"),
        feature_domain_sizes={"x": 3, "y": 2},
        target_assignments={},
        feature_priors={"x": (0.7, 0.2, 0.1), "y": (0.4, 0.6)},
        legal_actions=actions,
        action_priors={
            AssignValueAction("x", 0): 0.7,
            AssignValueAction("x", 1): 0.2,
            AddClusterAction("cluster_a"): 0.1,
        },
        cluster_candidates=(ClusterCandidate(key="cluster_a", colors=("x", "y")),),
        value_estimate=0.25,
    )


def test_trainable_policy_uses_model_scores():
    state = _sample_state()
    policy = TrainablePolicyValue(model=DummyModel())
    priors = policy.action_priors(state, list(state.legal_actions))

    assert priors[AssignValueAction("x", 1)] > priors[AssignValueAction("x", 0)]
    assert priors[AssignValueAction("x", 1)] > priors[AddClusterAction("cluster_a")]
    assert abs(sum(priors.values()) - 1.0) < 1e-9
    assert policy.value(state) == 0.8


def test_trainable_policy_can_blend_with_heuristic():
    state = _sample_state()
    heuristic = HeuristicPolicyValue()
    heuristic_priors = heuristic.action_priors(state, list(state.legal_actions))
    policy = TrainablePolicyValue(
        model=DummyModel(),
        fallback_policy=heuristic,
        policy_mix=0.5,
        value_mix=0.5,
    )

    blended_priors = policy.action_priors(state, list(state.legal_actions))
    assert blended_priors != heuristic_priors
    assert abs(sum(blended_priors.values()) - 1.0) < 1e-9
    assert policy.value(state) == 0.525


def test_shared_policy_value_net_predicts_scores_and_value():
    state = _sample_state()
    model = AlphaReasonPolicyValueNet(num_features=2, max_feature_dim=3, hidden_dim=64, action_hidden_dim=32)
    output = model.predict(state, list(state.legal_actions))

    assert output.action_scores is not None
    assert set(output.action_scores.keys()) == {action_to_key(action) for action in state.legal_actions}
    assert output.value is not None
    assert -1.0 <= output.value <= 1.0

