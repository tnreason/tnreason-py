from ..policies import HeuristicPolicyValue
from ..run_experiment import solve_task
from ..task import AlphaReasonTask


def test_solve_task_runs_through_policy_injection(monkeypatch):
    task = AlphaReasonTask(
        name="toy",
        canetwork_factory=lambda evidence: None,
        initial_evidence={},
        target_feature_keys=("x",),
        feature_domain_sizes={"x": 2},
        assignment_evidence_map={"x": {0: {"x": 0}, 1: {"x": 1}}},
    )
    captured = {}

    class FakeEngine:
        def cache_info(self):
            return {"hits": 0}

    class FakeEnv:
        def __init__(self):
            self.calls = 0

        def reset(self):
            captured["reset"] = True
            class _State:
                done = True
                assigned_count = 0
            return _State()

    def fake_engine_ctor():
        return FakeEngine()

    def fake_env_ctor(task, closure_engine):
        captured["task"] = task
        captured["closure_engine"] = closure_engine
        return FakeEnv()

    def fake_mcts_ctor(env, policy_value, simulations):
        captured["policy_value"] = policy_value
        captured["simulations"] = simulations
        class _Mcts:
            pass
        return _Mcts()

    monkeypatch.setattr("experiments.alphareason.run_experiment.AlphaReasonClosureEngine", fake_engine_ctor)
    monkeypatch.setattr("experiments.alphareason.run_experiment.AlphaReasonEnv", fake_env_ctor)
    monkeypatch.setattr("experiments.alphareason.run_experiment.AlphaReasonMCTS", fake_mcts_ctor)

    state, history, cache_info = solve_task(task, policy_value=HeuristicPolicyValue(), simulations=7)
    assert captured["task"] is task
    assert captured["simulations"] == 7
    assert history == []
    assert cache_info == {"hits": 0}
