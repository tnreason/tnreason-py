from ..policies import HeuristicPolicyValue
from ..run_experiment import count_correct_target_assignments, solution_assignments_from_evidence, solve_task
from ..state import AlphaReasonState
from ..task import AlphaReasonTask


def test_solve_task_runs_through_policy_injection(monkeypatch):
    task = AlphaReasonTask(
        name="toy",
        network_factory=lambda evidence: None,
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

    def fake_mcts_ctor(env, policy_value, simulations, rollout_depth=0):
        captured["policy_value"] = policy_value
        captured["simulations"] = simulations
        captured["rollout_depth"] = rollout_depth
        class _Mcts:
            pass
        return _Mcts()

    monkeypatch.setattr("experiments.alphareason.run_experiment.AlphaReasonClosureEngine", fake_engine_ctor)
    monkeypatch.setattr("experiments.alphareason.run_experiment.AlphaReasonEnv", fake_env_ctor)
    monkeypatch.setattr("experiments.alphareason.run_experiment.AlphaReasonMCTS", fake_mcts_ctor)

    state, history, cache_info = solve_task(task, policy_value=HeuristicPolicyValue(), simulations=7)
    assert captured["task"] is task
    assert captured["simulations"] == 7
    assert captured["rollout_depth"] == 0
    assert history == []
    assert cache_info == {"hits": 0}


def test_count_correct_target_assignments():
    task = AlphaReasonTask(
        name="toy",
        network_factory=lambda evidence: None,
        initial_evidence={},
        target_feature_keys=("x", "y", "z"),
        feature_domain_sizes={"x": 2, "y": 2, "z": 2},
        assignment_evidence_map={},
        solution_assignments={"x": 1, "y": 0, "z": 1},
    )
    state = AlphaReasonState(
        task_name="toy",
        evidence={},
        target_feature_keys=task.target_feature_keys,
        feature_domain_sizes=task.feature_domain_sizes,
        target_assignments={"x": 1, "y": 1},
    )

    assert count_correct_target_assignments(task, state) == (1, 2)


def test_solution_assignments_from_evidence():
    evidence = {
        "a_0_0_0_0_1": 1,
        "a_0_0_0_1_3": 1,
    }

    assert solution_assignments_from_evidence(
        evidence,
        ("pos_0_0_0_0", "pos_0_0_0_1", "pos_0_0_0_2"),
    ) == {
        "pos_0_0_0_0": 1,
        "pos_0_0_0_1": 3,
    }
