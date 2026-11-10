from itertools import product
from types import SimpleNamespace
import numpy as np
import pytest
from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from experiments.cluster_selection.candidate_evaluation_mac import (
    contract_candidate, evaluate_candidate, prepare_candidate)
from experiments.cluster_selection.candidate_generation import ClusterCandidateGenerator
from experiments.cluster_selection.data import network_data
from experiments.cluster_selection import runtime


def _state():
    relation = np.ones((16, 16), dtype=bool)
    relation[3] = False
    relation[3, 2] = True
    network = ConstraintNetwork({
        "ab": NumpyCore(values=relation.copy(), colors=["a", "b"]),
        "ac": NumpyCore(values=relation.copy(), colors=["a", "c"]),
        "bc": NumpyCore(values=~np.eye(16, dtype=bool), colors=["b", "c"]),
        "ad": NumpyCore(values=np.eye(16, dtype=bool), colors=["a", "d"]),
        "q": NumpyCore(values=np.ones(16, dtype=bool), colors=["q"])}, maintain_indices=True)
    state = network.create_mac_state(dict(a=(3, 7), b=(2, 9), c=(2, 9), d=(3, 7)))
    assert state.propagate()
    return network, state, SummaryClusterCandidate("triangle", ("ab", "ac", "bc"), ("a",))


def _brute_support(state, candidate):
    colors = tuple(dict.fromkeys(color for key in candidate.core_keys for color in state._cores[key].colors))
    values = np.zeros(tuple(state.dimensions[color] for color in candidate.open_colors), dtype=bool)
    for row in product(*(state.domains[color] for color in colors)):
        assignment = dict(zip(colors, row))
        if all(state._cores[key][{color: assignment[color] for color in state._cores[key].colors}] != 0
               for key in candidate.core_keys):
            values[tuple(assignment[color] for color in candidate.open_colors)] = True
    return values


@pytest.mark.parametrize("outputs", [("a",), ("c", "a"), ()])
@pytest.mark.parametrize("signed", [False, True])
def test_compact_matches_dense_and_bruteforce_without_mutating_inputs(outputs, signed):
    _, state, candidate = _state()
    candidate = SummaryClusterCandidate(candidate.key, candidate.core_keys, outputs)
    if signed:
        for key in candidate.core_keys:
            state._cores[key].values = state._cores[key].values.astype(float) * -3
            del state._contractions[key]  # Exercise uncached support conversion too.
    domains = state.snapshot_domains()
    originals = {key: core.values.copy() for key, core in state._cores.items()}
    dense = prepare_candidate(state, candidate)
    compact = prepare_candidate(state, candidate, compact_domains=True)
    assert len(dense) == 2 and len(compact) == 3
    assert compact[0].work < dense[0].work
    expected = _brute_support(state, candidate)
    for prepared in (dense, compact):
        summary = contract_candidate(candidate, prepared)
        np.testing.assert_array_equal(summary.values, expected)
        assert summary.values.dtype == bool
        assert tuple(summary.colors) == outputs
    assert state.snapshot_domains() == domains
    for key, values in originals.items():
        np.testing.assert_array_equal(state._cores[key].values, values)


def test_compaction_allows_an_informative_cluster_under_actual_size_and_work_limits():
    _, state, candidate = _state()
    compact = prepare_candidate(state, candidate, compact_domains=True)
    limits = dict(max_work=compact[0].work, max_size=compact[0].max_size)
    before, statistics = state.snapshot_domains(), state.statistics
    with pytest.raises(ValueError, match="budget"):
        prepare_candidate(state, candidate, **limits)
    result = evaluate_candidate(state, candidate, compact_domains=True, **limits)
    assert result.domains["a"] == result.domains["d"] == (7,)
    assert result.removed_domain_values == 2 and not result.contradiction
    assert state.snapshot_domains() == before and state.statistics == statistics
    assert not state._temporary_core_keys and not state._added_core_keys
    assert compact[0].output_size == 16 and compact[0].max_size >= 16


def test_expanded_output_memory_is_budgeted_before_contraction(monkeypatch):
    _, state, candidate = _state()
    candidate = SummaryClusterCandidate(candidate.key, candidate.core_keys, ("a", "b"))
    monkeypatch.setattr(np, "einsum", lambda *a, **k: pytest.fail("Over-budget contraction"))
    with pytest.raises(ValueError, match="budget"):
        prepare_candidate(state, candidate, compact_domains=True, max_size=255)
    assert prepare_candidate(state, candidate, compact_domains=True, max_size=256)[0].output_size == 256


def test_domain_changes_and_parent_rollback_rebuild_slices_and_keep_only_pruning():
    network, state, candidate = _state()
    before, topology = state.snapshot_domains(), tuple(state._cores)
    predictor = SimpleNamespace(kind="construction", network=network_data(network),
        config=dict(compact_domains=True, max_work=1_000_000, max_size=100_000),
        propose=lambda domains: candidate)
    runtime.attach(state, predictor)
    policy = state.on_closure
    for _ in range(2):
        state.push_level()
        assert state.propagate()
        assert state.domains["a"] == state.domains["d"] == (7,)
        assert not policy.plans and tuple(state._cores) == topology
        assert not state._temporary_core_keys and not state._added_core_keys
        state.pop_level()
        assert state.snapshot_domains() == before
    assert policy.statistics["useful"] == 2
    # The same domains after rollback reconstruct the removed values correctly.
    dense = contract_candidate(candidate, prepare_candidate(state, candidate))
    compact = contract_candidate(candidate, prepare_candidate(state, candidate, compact_domains=True))
    np.testing.assert_array_equal(compact.values, dense.values)


def test_scalar_contradiction_is_boolean_and_reversible():
    cores = {key: NumpyCore(values=~np.eye(4, dtype=bool), colors=list(key)) for key in ("ab", "bc", "ca")}
    state = ConstraintNetwork(cores).create_mac_state({color: (1, 3) for color in "abc"})
    assert state.propagate()
    candidate = SummaryClusterCandidate("odd", tuple(cores), ())
    before = state.snapshot_domains()
    state.push_level()
    assert runtime.apply_candidate(state, candidate, compact_domains=True).contradiction
    assert state.contradiction and not state._temporary_core_keys and not state._added_core_keys
    assert not any(key.startswith("summary_") for key in state._cores)
    state.pop_level()
    assert not state.contradiction and state.snapshot_domains() == before


def test_compact_noop_cache_reconsiders_changed_internal_domains_and_backtracking():
    network, state, candidate = _state()
    predictor = SimpleNamespace(kind="construction", network=network_data(network),
        config=dict(compact_domains=True, max_work=1_000_000, max_size=100_000),
        propose=lambda domains: candidate)
    runtime.attach(state, predictor)
    assert state.propagate()
    policy = state.on_closure
    assert policy.statistics["selected"] == 2  # Useful contraction, then completed no-op.
    state.push_level()
    state.restrict_domain("q", (0, 1))
    assert state.propagate() and policy.statistics["selected"] == 2
    state.assign("b", 2)
    assert state.propagate() and policy.statistics["selected"] == 3
    state.pop_level()
    assert state.propagate() and policy.statistics["selected"] == 4
    assert not policy.plans


def test_runtime_budget_stops_before_an_unaffordable_next_contraction():
    network, state, candidate = _state()
    budget = prepare_candidate(state, candidate, compact_domains=True)[0].work
    predictor = SimpleNamespace(kind="construction", network=network_data(network),
        config=dict(compact_domains=True, max_total_work=budget, max_work=1_000_000, max_size=100_000),
        propose=lambda domains: candidate)
    runtime.attach(state, predictor)
    assert state.propagate()
    assert state.domains["a"] == state.domains["d"] == (7,)
    assert state.on_closure.statistics["selected"] == 1
    assert state.on_closure.statistics["estimated_work"] == budget
    assert state.on_closure.statistics["budget_exhausted"] == 1


def test_compact_generator_budget_matches_runtime_and_does_not_reuse_dense_costs():
    network, state, candidate = _state()
    budget = prepare_candidate(state, candidate, compact_domains=True)[0].work
    generator = ClusterCandidateGenerator(network, max_work=budget, max_size=16, max_open_colors=1)
    assert not generator.generate(state.domains, cyclic_only=True)
    candidates = generator.generate(state.domains, cyclic_only=True, compact_domains=True)
    assert any(set(item.core_keys) == set(candidate.core_keys) for item in candidates)
    assert not generator.generate(state.domains, cyclic_only=True)
    for item in candidates:
        prepare_candidate(state, item, compact_domains=True, max_work=budget, max_size=16)


def test_compact_generator_retains_large_clusters_with_an_affordable_output_subset():
    network, state, candidate = _state()
    generator = ClusterCandidateGenerator(network, max_work=1000, max_size=256, max_open_colors=3)
    assert not generator.generate(state.domains, cyclic_only=True)
    proposals = generator.generate(state.domains, cyclic_only=True, compact_domains=True)
    chosen = next(item for item in proposals if set(item.core_keys) == set(candidate.core_keys))
    assert len(chosen.open_colors) == 2  # Three full axes need 4096 entries; two fit 256.
    plan = prepare_candidate(state, chosen, compact_domains=True, max_work=1000, max_size=256)
    assert plan[0].output_size == 256
    assert not generator.generate(state.domains, cyclic_only=True)  # Legacy proposals are unchanged.
