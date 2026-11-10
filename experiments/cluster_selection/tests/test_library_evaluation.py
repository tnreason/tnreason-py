from copy import deepcopy
from math import log

import numpy as np
import pytest

from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.cluster_selection.candidate_evaluation_mac import evaluate_candidate
from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from experiments.cluster_selection.data import network_data
from experiments.cluster_selection.library_evaluation import LibraryEvaluator, prefix_metrics, solve_library


def _library(cores, candidates, primary=None):
    raw = network_data(ConstraintNetwork(cores, maintain_indices=True))
    keys, colors = [core["key"] for core in raw["cores"]], raw["colors"]
    return dict(network=raw, reward_colors=colors if primary is None else primary,
                clusters=[dict(cores=[keys.index(key) for key in selected],
                               open=[colors.index(color) for color in opened])
                          for selected, opened in candidates])


def _triangle():
    relation = np.array([[1, 0, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
    return {"ap": NumpyCore(values=relation, colors=["a", "p"]),
            "aq": NumpyCore(values=relation, colors=["a", "q"]),
            "pq": NumpyCore(values=~np.eye(3, dtype=bool), colors=["p", "q"])}


def _record(evaluator, domains):
    return dict(masks=[sum(1 << value for value in domains[color]) for color in evaluator.colors])


def _snapshot(evaluator):
    state = evaluator.state
    return (state.snapshot_domains(), state.statistics, tuple(state._trail), state.decision_level,
            tuple(state._cores), tuple(state._temporary_core_keys), frozenset(state._added_core_keys),
            tuple(state._queue), frozenset(state._queued), state.contradiction,
            tuple(tuple(state._domain_tensors[color]) for color in evaluator.colors))


def _assert_full_recomputation(evaluator, candidate):
    state, before = evaluator.state, evaluator.state.snapshot_domains()
    snapshot = _snapshot(evaluator)
    expected = evaluate_candidate(state, candidate, cost_weight=0.)
    actual = evaluator.measure(candidate)
    assert _snapshot(evaluator) == snapshot
    assert actual["contradiction"] == expected.contradiction
    assert actual["removed_values"] == expected.removed_domain_values
    assert actual["primary_removed_values"] == sum(
        len(before[color]) - len(expected.domains[color]) for color in evaluator.primary)
    for name, weights in (("gain", evaluator.weights),
                          ("primary_gain", {color: 1. for color in evaluator.primary})):
        remaining = sum(weight * log(len(before[color])) for color, weight in weights.items())
        gain = 1. if expected.contradiction else (
            sum(weight * log(len(before[color]) / len(expected.domains[color]))
                for color, weight in weights.items()) / remaining if remaining else 0.)
        assert actual[name] == pytest.approx(gain)
    return actual


def test_entropy_delta_counts_first_domain_before_repeated_revisions(monkeypatch):
    cores = _triangle()
    cores.update(ab=NumpyCore(values=np.array([[1, 1, 1], [0, 1, 1], [0, 1, 1]], dtype=bool),
                             colors=["a", "b"]),
                 bc=NumpyCore(values=np.eye(3, dtype=bool), colors=["b", "c"]),
                 ca=NumpyCore(values=np.array([[1, 1, 0], [1, 0, 1], [1, 0, 1]], dtype=bool),
                             colors=["c", "a"]))
    evaluator = LibraryEvaluator(_library(cores, [(tuple(_triangle()), ("a",))], ["a"]))
    domains = {color: (0, 1) if color in ("p", "q") else (0, 1, 2) for color in evaluator.colors}
    with evaluator.closed(_record(evaluator, domains)):
        revisions, original = [], evaluator.state._set_domain

        def revise(color, values, source=None):
            if color == "a" and values != evaluator.state.domains[color]:
                revisions.append(tuple(values))
            return original(color, values, source)

        monkeypatch.setattr(evaluator.state, "_set_domain", revise)
        outcome = _assert_full_recomputation(evaluator, evaluator.candidates[0])
        assert revisions[:2] == [(1, 2), (2,)]
        assert outcome["primary_gain"] == 1. and outcome["primary_removed_values"] == 2
        assert 0. < outcome["gain"] < 1.


@pytest.mark.parametrize("outputs", [(), ("a",)])
def test_scalar_and_nonscalar_contradictions_restore_parent_state(outputs):
    cores = {key: NumpyCore(values=~np.eye(3, dtype=bool), colors=list(key))
             for key in ("ab", "bc", "ca")}
    evaluator = LibraryEvaluator(_library(cores, [(tuple(cores), outputs)]))
    before = _snapshot(evaluator)
    record = _record(evaluator, {color: (0, 1) for color in evaluator.colors})
    with evaluator.closed(record):
        outcome = _assert_full_recomputation(evaluator, evaluator.candidates[0])
        assert outcome["contradiction"] and outcome["gain"] == outcome["primary_gain"] == 1.
        assert bool(outcome["removed_values"]) == bool(outputs)
    assert _snapshot(evaluator) == before
    rows = evaluator.score_state(record)["successes"]
    assert len(rows) == 1 and rows[0]["contradiction"]
    assert _snapshot(evaluator) == before


def test_reused_plans_follow_current_masks_and_trials_are_independent():
    cores = _triangle()
    library = _library(cores, [(tuple(cores), ("a",)), (tuple(cores), ("a", "p"))])
    saved = deepcopy(library)
    evaluator = LibraryEvaluator(library)
    before = _snapshot(evaluator)
    strong = _record(evaluator, {"a": (0, 1, 2), "p": (0, 1), "q": (0, 1)})
    weak = _record(evaluator, {color: (0, 1, 2) for color in evaluator.colors})
    first = evaluator.score_state(strong, sizes=[0, 1, 2])
    plans = tuple(map(id, evaluator.plans.values()))
    assert [row["index"] for row in first["successes"]] == [0, 1]
    assert len({row["gain"] for row in first["successes"]}) == 1
    assert not evaluator.score_state(weak)["successes"]
    assert tuple(map(id, evaluator.plans.values())) == plans
    again = evaluator.score_state(strong, sizes=[2, 1, 0, 2])
    assert first["successes"] == again["successes"] and first["prefixes"] == again["prefixes"]
    assert _snapshot(evaluator) == before and library == saved


def test_reference_candidates_use_same_closed_domains_without_mutating_library(monkeypatch):
    from experiments.cluster_selection import library_evaluation

    cores = _triangle()
    evaluator = LibraryEvaluator(_library(cores, [(tuple(cores), ("a",))]))
    expected = {"a": (0, 1, 2), "p": (0, 1), "q": (0, 1)}
    calls = []

    def generate(self, domains, *, cyclic_only):
        calls.append(dict(domains))
        assert dict(domains) == expected
        assert cyclic_only is True
        return [SummaryClusterCandidate("other", tuple(cores), ("a",))] * 2

    monkeypatch.setattr(library_evaluation.ClusterCandidateGenerator, "generate", generate)
    result = evaluator.score_state(_record(evaluator, expected), reference_candidates=2)
    assert calls == [expected]
    reference = result["reference"]
    assert reference["proposed"] == reference["evaluated"] == reference["useful"] == 2
    assert reference["best_gain"] == result["successes"][0]["gain"]


def test_affordability_rejection_is_recorded_separately_from_noop():
    cores = _triangle()
    evaluator = LibraryEvaluator(_library(cores, [(tuple(cores), ("a",))]), max_work=1)
    before = _snapshot(evaluator)
    result = evaluator.score_state(dict(domains={color: (0, 1, 2) for color in evaluator.colors}))
    assert result["rejected"] == [0] and result["evaluated"] == 0 and not result["successes"]
    assert _snapshot(evaluator) == before


@pytest.mark.parametrize("record", [dict(masks=[0, 3]), dict(masks=[4, 3]),
                                         dict(masks=[True, 3]), dict(masks=[3]),
                                         dict(domains=[(0,), (0, 1)]),
                                         dict(domains=[(0,), (1,)])])
def test_invalid_or_nonclosed_states_restore_base(record):
    evaluator = LibraryEvaluator(_library(
        {"ab": NumpyCore(values=np.eye(2, dtype=bool), colors=["a", "b"])},
        [(('ab',), ('a',))]))
    before = _snapshot(evaluator)
    with pytest.raises(ValueError):
        evaluator.score_state(record)
    assert _snapshot(evaluator) == before


@pytest.mark.parametrize("sizes", [[-1], [2], [True], [1.5]])
def test_prefix_bounds_rejected(sizes):
    evaluator = LibraryEvaluator(_library(_triangle(), [(tuple(_triangle()), ("a",))]))
    with pytest.raises(ValueError, match="Prefix"):
        evaluator.score_state(dict(masks=[7, 7, 7]), sizes=sizes)


def test_nested_prefixes_keep_library_indices_and_monotone_best_gain():
    successes = [dict(index=1, gain=.8, primary_gain=.1), dict(index=3, gain=.2, primary_gain=.9)]
    result = prefix_metrics(successes, [0, 1, 2, 3, 4])
    assert [row["successes"] for row in result] == [0, 0, 1, 1, 2]
    assert [row["best_gain"] for row in result] == [0., 0., .8, .8, .8]
    assert [row["best_primary_gain"] for row in result] == [0., 0., .1, .1, .9]


def test_sparse_entropy_matches_full_recomputation_on_random_generic_closures():
    rng = np.random.default_rng(12)
    for _ in range(24):
        cores = {}
        for key in ("ab", "bc", "ca", "ad", "cd"):
            values = rng.random((3, 3)) < .55
            values[0, 0] = True  # A planted assignment keeps every sampled closure satisfiable.
            cores[key] = NumpyCore(values=values, colors=list(key))
        candidates = [(tuple(cores), outputs) for outputs in ((), ("a",), ("b", "d"))]
        evaluator = LibraryEvaluator(_library(cores, candidates, ["a"]))
        initial = {color: tuple(sorted({0, *np.flatnonzero(rng.random(3) < .6)}))
                   for color in evaluator.colors}
        state = evaluator.network.create_mac_state(initial)
        assert state.propagate()
        before = _snapshot(evaluator)
        with evaluator.closed(_record(evaluator, state.domains)):
            for candidate in evaluator.candidates:
                _assert_full_recomputation(evaluator, candidate)
        assert _snapshot(evaluator) == before


def _binary_triangle(prefix=""):
    rename = lambda colors: [prefix + color for color in colors]
    return {prefix + "ab": NumpyCore(values=np.eye(2, dtype=bool), colors=rename("ab")),
            prefix + "bc": NumpyCore(values=np.eye(2, dtype=bool), colors=rename("bc")),
            prefix + "ac": NumpyCore(values=np.array([[1, 1], [1, 0]], dtype=bool),
                                     colors=rename("ac"))}


def test_library_oracle_solves_by_pruning_and_counts_actual_contractions():
    cores = _binary_triangle()
    library = _library(cores, [(tuple(cores), ("a",))])
    outcome = solve_library(library, {})
    assert outcome["status"] == "solved" and outcome["valid"] and outcome["nodes"] == 1
    assert outcome["accepted_applications"] == outcome["application_contractions"] == 1
    assert outcome["oracle_measurements"] == 1 and outcome["oracle_contractions"] == 2
    assert outcome["contractions"] == 3 and outcome["rejected"] == 0


@pytest.mark.parametrize("ineffective", [False, True])
def test_empty_or_ineffective_library_preserves_plain_mac_solving(ineffective):
    cores = _binary_triangle()
    library = _library(cores, [(('ab',), ('a',))] if ineffective else [])
    network = ConstraintNetwork(cores, maintain_indices=True)
    expected = network.solve_mac()
    outcome = solve_library(library, {})
    assert outcome["nodes"] == expected.nodes and outcome["solved"] == expected.solved
    assert outcome["status"] == "solved" and outcome["valid"]
    assert outcome["accepted_applications"] == outcome["application_contractions"] == 0
    assert outcome["oracle_contractions"] == outcome["oracle_measurements"]


def test_library_oracle_allowance_is_sequential_useful_application_cap():
    first, second = _binary_triangle(), _binary_triangle("other_")
    library = _library({**first, **second}, [(tuple(first), ("a",)),
                                           (tuple(second), ("other_a",))])
    limited = solve_library(library, {}, allowance=1)
    broader = solve_library(library, {}, allowance=2)
    assert limited["valid"] and broader["valid"]
    assert limited["accepted_applications"] == 1 and limited["nodes"] == 2
    assert broader["accepted_applications"] == 2 and broader["nodes"] == 1


def test_library_oracle_contradiction_cleans_state_and_hook(monkeypatch):
    from experiments.cluster_selection import library_evaluation

    cores = {key: NumpyCore(values=~np.eye(3, dtype=bool), colors=list(key))
             for key in ("ab", "bc", "ca")}
    library, engines = _library(cores, [(tuple(cores), ())]), []

    def capture(*args, **kwargs):
        engines.append(LibraryEvaluator(*args, **kwargs))
        return engines[-1]

    monkeypatch.setattr(library_evaluation, "LibraryEvaluator", capture)
    outcome = solve_library(library, {color: (0, 1) for color in "abc"})
    state = engines[0].state
    assert outcome["status"] == "unsat" and outcome["unsatisfiable"] and outcome["nodes"] == 1
    assert outcome["accepted_applications"] == 1 and not outcome["valid"]
    assert state.decision_level == 0 and state.on_closure is None
    assert not state._temporary_core_keys and not state._added_core_keys
    assert set(state._cores) == set(cores)
    assert all(tuple(np.flatnonzero(state._domain_tensors[color])) == values
               for color, values in state.domains.items())


def test_library_oracle_node_limit_cleans_branch_state(monkeypatch):
    from experiments.cluster_selection import library_evaluation

    cores = _binary_triangle()
    library, engines = _library(cores, []), []

    def capture(*args, **kwargs):
        engines.append(LibraryEvaluator(*args, **kwargs))
        return engines[-1]

    monkeypatch.setattr(library_evaluation, "LibraryEvaluator", capture)
    outcome = solve_library(library, {}, nodes=1)
    assert outcome["status"] == "node_limit" and outcome["nodes"] == 1
    assert not outcome["solved"] and not outcome["unsatisfiable"]
    state = engines[0].state
    assert state.decision_level == 0 and state.on_closure is None
    assert not state._temporary_core_keys and not state._added_core_keys
    assert not state._queue and not state._queued
