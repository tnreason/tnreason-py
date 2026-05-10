from .policies import HeuristicPolicyValue, TrainablePolicyValue
from .run_experiment import solve_puzzle


class DummyModel:
    def predict(self, state, legal_actions):
        raise AssertionError("This test only verifies wiring, not model execution.")


def test_solve_puzzle_forwards_policy_value(monkeypatch):
    captured = {}
    policy = TrainablePolicyValue(model=DummyModel(), fallback_policy=HeuristicPolicyValue())

    def fake_solve_with_mcts(**kwargs):
        captured.update(kwargs)
        return "sentinel"

    monkeypatch.setattr(
        "experiments.alphasudoku.run_experiment._solve_with_mcts",
        fake_solve_with_mcts,
    )

    result = solve_puzzle(
        canetwork_factory=object(),
        challenge_puzzle_num=7,
        policy_value=policy,
        num=3,
        max_steps=5,
        simulations=11,
    )

    assert result == "sentinel"
    assert captured["challenge_puzzle_num"] == 7
    assert captured["policy_value"] is policy
