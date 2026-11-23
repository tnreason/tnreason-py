import numpy as np
import pytest
from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate, evaluate_summary_cluster_candidate
from experiments.cluster_selection.candidate_generation import ClusterCandidateGenerator
from experiments.cluster_selection.candidate_evaluation_mac import evaluate_candidate
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.sudoku_tests.standard_constraints import get_sudoku_constraint_network


def _state(signed=False):
    relation = np.array([[1, 0, 1], [1, 1, 1]])
    cores = {"ab": NumpyCore(values=relation, colors=["a", "b"]),
             "ac": NumpyCore(values=relation, colors=["a", "c"]),
             "bc": NumpyCore(values=~np.eye(3, dtype=bool), colors=["b", "c"]),
             "ad": NumpyCore(values=np.eye(2, dtype=bool), colors=["a", "d"])}
    if signed:
        for core in cores.values():
            core.values = core.values.astype(float) * np.where(np.indices(core.shape).sum(axis=0) % 2, -2, 3)
    state = ConstraintNetwork(cores).create_mac_state({"b": (0, 1), "c": (0, 1)})
    assert state.propagate()
    return state, SummaryClusterCandidate("project", ("ab", "ac", "bc"), ("a",))


def _snapshot(state):
    return (state.snapshot_domains(), state.statistics, state.decision_level,
            {color: (id(tensor), tuple(tensor)) for color, tensor in state._domain_tensors.items()},
            {key: id(core) for key, core in state._cores.items()},
            {key: (id(cache), id(cache[0])) for key, cache in state._contractions.items()},
            dict(state._core_keys_by_color), tuple(state._queue), frozenset(state._queued),
            tuple(state._trail), tuple(state._level_markers), tuple(state._temporary_core_keys),
            frozenset(state._added_core_keys), frozenset(state._changed_colors), state.on_closure,
            state.contradiction, state.contradiction_reason)


@pytest.mark.parametrize("signed", [False, True])
def test_domains_enter_the_contraction_and_pruning_cascades_then_rolls_back(signed):
    state, candidate = _state(signed)
    before = _snapshot(state)
    if not signed:
        old = evaluate_summary_cluster_candidate(state._cores, state.domains, candidate)
        assert old.domains["a"] == (0, 1)  # Domain-free projection misses this implication.
    result = evaluate_candidate(state, candidate, cost_weight=2)
    assert result.domains["a"] == result.domains["d"] == (1,)
    assert result.removed_domain_values == 2 and not result.contradiction
    assert result.propagation.removed_values == 2
    assert result.cost >= 0 and result.reward == pytest.approx(2 - 2 * result.cost)
    assert _snapshot(state) == before
    assert evaluate_candidate(state, candidate).domains == result.domains
    assert _snapshot(state) == before


def test_existing_search_levels_hooks_history_and_counters_are_preserved():
    state, candidate = _state()
    state.push_level()
    state.on_closure = lambda *args: pytest.fail("Other cluster hooks must not run during a trial")
    state._added_core_keys.add("summary_project")
    before = _snapshot(state)
    result = evaluate_candidate(state, candidate, reward_colors=("a", "a"))
    assert result.removed_domain_values == 1
    assert result.propagation.removed_values == 2
    assert _snapshot(state) == before
    state.on_closure = None
    state.pop_level()


def test_the_estimated_path_executes_with_persistent_boolean_inputs(monkeypatch):
    state, candidate = _state()
    original, calls = np.einsum, []

    def capture(*args, **kwargs):
        calls.append(kwargs.get("optimize"))
        if "optimize" in kwargs:
            arrays = args[:-1:2]
            assert all(array.dtype == bool for array in arrays)
            for key in candidate.core_keys:
                assert any(array is state._contractions[key][0] for array in arrays)
            for color in ("a", "b", "c"):
                assert any(array is state._domain_tensors[color] for array in arrays)
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "einsum", capture)
    result = evaluate_candidate(state, candidate)
    assert ["einsum_path", *result.estimate.path] in calls


@pytest.mark.parametrize("stage", ["contraction", "propagation"])
def test_exceptions_restore_the_starting_state(monkeypatch, stage):
    state, candidate = _state()
    state.on_closure = lambda *args: {}
    before = _snapshot(state)

    def fail(*args, **kwargs):
        if stage == "propagation":
            state.restrict_domain("a", (1,))
        raise RuntimeError("injected trial failure")

    monkeypatch.setattr(np if stage == "contraction" else state,
                        "einsum" if stage == "contraction" else "propagate", fail)
    with pytest.raises(RuntimeError, match="injected"):
        evaluate_candidate(state, candidate)
    assert _snapshot(state) == before


@pytest.mark.parametrize("budget", [{"max_work": 0}, {"max_size": 1}])
def test_budgets_reject_before_contraction_without_changing_state(monkeypatch, budget):
    state, candidate = _state()
    before = _snapshot(state)
    monkeypatch.setattr(np, "einsum", lambda *args, **kwargs: pytest.fail("Over-budget contraction executed"))
    with pytest.raises(ValueError, match="budget"):
        evaluate_candidate(state, candidate, **budget)
    assert _snapshot(state) == before


def test_a_pending_gac_queue_is_rejected_without_repeating_closure():
    state, candidate = _state()
    state.restrict_domain("b", (0,))
    before = _snapshot(state)
    with pytest.raises(ValueError, match="closure"):
        evaluate_candidate(state, candidate)
    assert _snapshot(state) == before


def test_scalar_contradiction_bonus_and_restoration():
    cores = {key: NumpyCore(values=~np.eye(3, dtype=bool), colors=list(key)) for key in ("ab", "bc", "ca")}
    state = ConstraintNetwork(cores).create_mac_state({color: (0, 1) for color in "abc"})
    assert state.propagate()
    before = _snapshot(state)
    result = evaluate_candidate(state, SummaryClusterCandidate("odd", tuple(cores), ()), contradiction_reward=7)
    assert result.contradiction and result.reward == pytest.approx(7 - result.cost)
    assert result.estimate.output_size == 1 and _snapshot(state) == before


def test_many_supporting_assignments_cannot_overflow():
    state = ConstraintNetwork({"ax": NumpyCore(values=np.ones((2, 256), dtype=np.uint8),
                                              colors=["a", "x"])}).create_mac_state()
    assert state.propagate()
    result = evaluate_candidate(state, SummaryClusterCandidate("many", ("ax",), ("a",)))
    assert not result.contradiction and result.domains["a"] == (0, 1)


def test_uncached_core_support_is_converted_without_mutating_the_core():
    state, candidate = _state(signed=True)
    del state._contractions["ab"]
    before = _snapshot(state)
    assert evaluate_candidate(state, candidate).domains["a"] == (1,)
    assert _snapshot(state) == before


@pytest.mark.parametrize("kind", ["duplicate", "missing", "open", "reward", "active_summary"])
def test_invalid_candidates_leave_the_state_unchanged(kind):
    state, candidate = _state()
    kwargs = {}
    if kind == "duplicate":
        candidate = SummaryClusterCandidate("bad", ("ab", "ab"), ("a",))
    elif kind == "missing":
        candidate = SummaryClusterCandidate("bad", ("missing",), ("a",))
    elif kind == "open":
        candidate = SummaryClusterCandidate("bad", ("ab",), ("missing",))
    elif kind == "reward":
        kwargs["reward_colors"] = ("missing",)
    else:
        candidate = SummaryClusterCandidate("project", ("ab",), ("a",))
        state.add_cores({"summary_project": NumpyCore(values=np.ones(2, dtype=bool), colors=["a"])})
        assert state.propagate()
    before = _snapshot(state)
    with pytest.raises((ValueError, KeyError)):
        evaluate_candidate(state, candidate, **kwargs)
    assert _snapshot(state) == before


def test_generated_hidden_pair_on_a_full_nine_by_nine_network():
    positions, units = ("pos_0_0_0_0", "pos_0_0_1_0"), ("row_0_0_0", "row_1_0_0")
    network = ConstraintNetwork(get_sudoku_constraint_network(num=3))
    domains = {f"pos_0_0_{col // 3}_{col % 3}": (0, 1, 2) if col in (0, 3) else tuple(range(2, 9))
               for col in range(9)}
    state = network.create_mac_state(domains)
    assert state.propagate()
    before = _snapshot(state)
    candidate = next(candidate for candidate in ClusterCandidateGenerator(network).generate(state.domains)
                     if len(candidate.core_keys) == 8 and set(candidate.open_colors) == set(positions + units))
    result = evaluate_candidate(state, candidate, reward_colors=positions)
    assert result.removed_domain_values == 2 and not result.contradiction
    assert all(result.domains[position] == (0, 1) for position in positions)
    assert _snapshot(state) == before
