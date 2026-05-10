from ..evaluate import summarize_state
from ..state import AlphaReasonState
from ..task import AlphaReasonTask


def test_summarize_state_counts_correct_assignments():
    task = AlphaReasonTask(
        name="toy",
        network_factory=lambda evidence: None,
        initial_evidence={},
        target_feature_keys=("x", "y"),
        feature_domain_sizes={"x": 2, "y": 2},
        assignment_evidence_map={"x": {0: {"x": 0}}, "y": {1: {"y": 1}}},
        solution_assignments={"x": 0, "y": 1},
    )
    state = AlphaReasonState(
        task_name="toy",
        evidence={},
        target_feature_keys=("x", "y"),
        feature_domain_sizes={"x": 2, "y": 2},
        target_assignments={"x": 0, "y": 0},
        solved=False,
        contradiction=False,
        assigned_count=2,
    )

    result = summarize_state("toy", task, state, history=[1, 2], cache_info={"hits": 3, "misses": 4})
    assert result.correct_assigned_count == 1
    assert not result.exact_match
    assert result.history_length == 2
    assert result.cache_hits == 3
