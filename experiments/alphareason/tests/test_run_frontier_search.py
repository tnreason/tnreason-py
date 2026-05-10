from experiments.alphareason.run_frontier_search import solve_task_with_frontier_search
from experiments.alphareason.task import AlphaReasonTask


def test_frontier_search_returns_root_if_done(monkeypatch):
    task = AlphaReasonTask(
        name="toy",
        network_factory=lambda evidence: None,
        initial_evidence={},
        target_feature_keys=("x",),
        feature_domain_sizes={"x": 2},
        assignment_evidence_map={"x": {0: {"x": 0}, 1: {"x": 1}}},
    )

    class FakeEngine:
        def cache_info(self):
            return {"hits": 0}

    class FakeEnv:
        def __init__(self, task, closure_engine):
            pass

        def reset(self):
            class _State:
                done = True
                solved = True
                contradiction = False
                assigned_count = 1
            return _State()

    monkeypatch.setattr("experiments.alphareason.run_frontier_search.AlphaReasonClosureEngine", FakeEngine)
    monkeypatch.setattr("experiments.alphareason.run_frontier_search.AlphaReasonEnv", FakeEnv)

    state, history, cache_info, diagnostics = solve_task_with_frontier_search(task)

    assert state.solved
    assert history == []
    assert cache_info == {"hits": 0}
    assert diagnostics.stop_reason == "root_done"
