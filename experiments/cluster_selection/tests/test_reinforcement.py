"""Construction policy, measured rewards, reversible trials and old-model compatibility."""
from copy import deepcopy
from itertools import islice, product
import json
from math import prod
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from experiments.cluster_selection.data import network_data
from experiments.cluster_selection import runtime

torch = pytest.importorskip("torch")
from experiments.cluster_selection.reinforcement.model import Policy, load
from experiments.cluster_selection.reinforcement import train


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _network(odd=False):
    relation = ~np.eye(3, dtype=bool) if odd else np.array([[1, 0, 1], [1, 1, 1], [1, 1, 1]])
    return ConstraintNetwork({"ab": NumpyCore(values=relation, colors=["a", "b"]),
        "ac": NumpyCore(values=relation, colors=["a", "c"]),
        "bc": NumpyCore(values=~np.eye(3, dtype=bool), colors=["b", "c"])}, maintain_indices=True)


def _data(odd=False):
    network = _network(odd)
    state = network.create_mac_state({color: (0, 1) for color in "abc"})
    assert state.propagate()
    return dict(version=1, network=network_data(network), reward_colors=["a"],
                samples=[dict(domains=list(state.domains.values()))])


def _snapshot(state):
    return (state.snapshot_domains(), state.statistics, tuple(state._cores), tuple(state._trail),
            state.decision_level, tuple(state._queue), frozenset(state._queued),
            frozenset(state._added_core_keys), state.on_closure)


def _checkpoint(path, model):
    torch.save(dict(version=1, kind=model.kind, network=model.network, config=model.config,
                    reward_colors=model.reward_colors, weights=model.state_dict(), beta=model.beta), path)


def test_frontier_grows_from_whole_cluster_and_outputs_include_internal_booleans():
    network = ConstraintNetwork({key: NumpyCore(values=np.ones((2, 2)), colors=list(key))
                                 for key in ("ab", "bc", "cd", "de")}, maintain_indices=True)
    model = Policy(network_data(network), width=8, rounds=1, max_size=4)
    assert ("core", 2) in model.actions([0, 1], [], False)  # Not adjacent to the seed ab.
    assert ("core", 0) not in model.actions([0, 1], [], False)
    assert ("core", 3) not in model.actions([0, 1], [], False)
    assert {idx for kind, idx in model.actions([0, 1], [], True) if kind == "open"} == {0, 1, 2}
    assert ("finish", -1) in model.actions([0, 1], [], True)  # Scalar output is legal.
    assert not any(kind == "open" for kind, _ in model.actions([0, 1, 2], [0, 1], True))


def test_sampled_rollouts_are_bounded_connected_and_use_one_encoding(monkeypatch):
    data = _data()
    model = Policy(data["network"], width=8, rounds=1, max_cores=3, max_open_colors=2)
    calls, original = [], model.encode
    monkeypatch.setattr(model, "encode", lambda domains: calls.append(1) or original(domains))
    with torch.no_grad():
        for _ in range(16):
            candidate, log_prob, entropy, trace = model.rollout(data["samples"][0]["domains"])
            keys, opened, phase = [], [], False
            for action in trace:
                assert action in model.actions(keys, opened, phase)
                if action[0] == "core":
                    keys.append(action[1])
                elif action[0] == "open":
                    opened.append(action[1])
                elif action[0] == "finish":
                    phase = True
            assert len(trace) <= 3 + 2 + 3
            assert torch.isfinite(log_prob) and torch.isfinite(entropy)
            if candidate:
                assert set(candidate.core_keys) == {model.keys[idx] for idx in keys}
                assert prod(data["network"]["dimensions"][idx] for idx in opened) <= model.config["max_size"]
    assert len(calls) == 16


def test_small_random_head_passes_gradients_and_beta_changes_preferences():
    torch.manual_seed(1)
    data = _data()
    model = Policy(data["network"], width=8, rounds=1)
    encoded = model.encode(data["samples"][0]["domains"])
    targets = model._targets(data["samples"][0]["domains"])
    actions, plain = model.distribution(encoded, [2, 2, 2], [0, 1], [], False, 0, targets)
    _, guided = model.distribution(encoded, [2, 2, 2], [0, 1], [], False, 3, targets)
    assert not torch.equal(plain.probs, guided.probs)
    assert (guided.probs > 0).all() and model.head[-1].weight.abs().sum() > 0
    (-guided.log_prob(torch.tensor(actions.index(("core", 2))))).backward()
    for module in (model.variable_encoder, model.core_encoder, model.head):
        assert module[0].weight.grad.abs().sum() > 0
        assert torch.isfinite(module[0].weight.grad).all()


def test_untrained_guidance_constructs_actual_useful_proposal_and_exact_outputs():
    from experiments.cluster_selection.candidate_generation import ClusterCandidateGenerator
    torch.manual_seed(0)
    data = _data()
    model = Policy(data["network"], width=8, rounds=1)
    model.reward_colors = data["reward_colors"]
    state = train._state(data)
    with train._closed(state, data["samples"][0], model.network["colors"]):
        before = _snapshot(state)
        proposals = ClusterCandidateGenerator(_network()).generate(state.domains, cyclic_only=True)
        candidate = model.propose(state.domains)
        assert candidate is not None
        assert (candidate.core_keys, candidate.open_colors) in [(p.core_keys, p.open_colors) for p in proposals]
        _, _, metrics = train.trial(model, state, greedy=True, cost_weight=.1)
        assert metrics["useful"] == 1 and metrics["gain"] == 1 and metrics["reward"] == .8
        assert _snapshot(state) == before


def test_prior_tracks_both_phases_without_excluding_off_heuristic_actions():
    model = Policy(_data()["network"], width=8, rounds=1)
    targets = ((frozenset((0, 1, 2)), frozenset((0,))),)
    actions = model.actions([0], [], False)
    prior = model._prior(actions, [0], [], False, targets)
    assert prior[actions.index(("core", 1))] > prior[actions.index(("finish", -1))]
    assert (prior.exp() > 0).all() and prior.exp().sum() == pytest.approx(1.)
    actions = model.actions([0, 1, 2], [], True)
    prior = model._prior(actions, [0, 1, 2], [], True, targets)
    assert prior.argmax() == actions.index(("open", 0))
    actions = model.actions([0, 1, 2], [0], True)
    assert model._prior(actions, [0, 1, 2], [0], True, targets).argmax() == actions.index(("finish", -1))
    # Once outside all target outputs/cores, only the learned logits guide choices.
    actions = model.actions([0, 1, 2], [1], True)
    assert not model._prior(actions, [0, 1, 2], [1], True, targets).any()
    assert not model._prior(model.actions([0], [], False), [0], [], False,
                            ((frozenset((1, 2)), frozenset((0,))),)).any()


def test_empty_heuristic_has_no_preference_not_the_old_local_scores():
    data = _data()
    model = Policy(data["network"], width=8, rounds=1)
    encoded = model.encode(data["samples"][0]["domains"])
    _, plain = model.distribution(encoded, [2, 2, 2], [0], [], False, 0)
    _, guided = model.distribution(encoded, [2, 2, 2], [0], [], False, 3)
    torch.testing.assert_close(plain.probs, guided.probs)


@pytest.mark.parametrize("beta,count", [(0., 32), (3., 0)])
def test_guidance_disabled_never_builds_generator(monkeypatch, beta, count):
    from experiments.cluster_selection.reinforcement import model as module
    data = _data()
    model = Policy(data["network"], width=8, rounds=1, heuristic_candidates=count)
    monkeypatch.setattr(module, "ClusterCandidateGenerator", lambda *a, **k: pytest.fail("Unexpected heuristic work"))
    with torch.no_grad():
        model.rollout(data["samples"][0]["domains"], beta=beta)
    assert model.generator is None and not model._targets_cache


def test_targets_cache_reuses_full_domain_values_and_is_bounded():
    model = Policy(_data()["network"], width=8, rounds=1)
    calls = []

    def generate(domains, **options):
        assert options == dict(cyclic_only=True)
        calls.append(tuple(domains.values()))
        return ()

    model.generator = SimpleNamespace(generate=generate)
    first = ((0, 1),) * 3
    assert model._targets(first) == model._targets(first) == () and len(calls) == 1
    model._targets(((0, 2), (0, 1), (0, 1)))  # Same sizes, different allowed values.
    assert len(calls) == 2
    values = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    for domains in islice(product(values, repeat=3), 130):
        model._targets(domains)
    assert len(model._targets_cache) == 128 and first not in model._targets_cache


@pytest.mark.parametrize("odd,opened", [(False, ("a",)), (True, ())])
def test_terminal_tensor_trial_retains_no_pruning_or_topology(monkeypatch, odd, opened):
    data = _data(odd)
    model = Policy(data["network"], width=8, rounds=1)
    model.reward_colors = data["reward_colors"]
    candidate = SummaryClusterCandidate("trial", ("ab", "ac", "bc"), opened)
    monkeypatch.setattr(model, "rollout", lambda *a, **k: (candidate, torch.tensor(0.), torch.tensor(0.), []))
    state = train._state(data)
    with train._closed(state, data["samples"][0], model.network["colors"]):
        before = _snapshot(state)
        _, _, metrics = train.trial(model, state, cost_weight=0, contradiction_reward=3)
        assert _snapshot(state) == before
        assert metrics["useful"] == 1 and metrics["contradictions"] == int(odd)
        assert metrics["treewidth"] == 2
        assert metrics["reward"] == metrics["gain"] + 3 * odd


def test_reward_uses_treewidth_not_elapsed_time_and_skip_is_zero(monkeypatch):
    data = _data()
    model = Policy(data["network"], width=8, rounds=1)
    model.reward_colors = ["a"]
    candidate = SummaryClusterCandidate("cost", ("ab",), ("a",))
    monkeypatch.setattr(model, "rollout", lambda *a, **k: (candidate, torch.tensor(0.), torch.tensor(0.), []))
    monkeypatch.setattr(train, "evaluate_candidate", lambda *a, **k: SimpleNamespace(
        removed_domain_values=2, contradiction=True, cost=.3))
    monkeypatch.setattr(train, "perf_counter", lambda: next(ticks))
    state = train._state(data)
    for seconds in (.5, 500.):
        ticks = iter([1., 1. + seconds])
        _, _, metrics = train.trial(model, state, cost_weight=2, contradiction_reward=5)
        assert metrics["reward"] == 5 and metrics["treewidth"] == 1
        assert metrics["cost_s"] == seconds
    monkeypatch.setattr(model, "rollout", lambda *a, **k: (None, torch.tensor(0.), torch.tensor(0.), []))
    ticks = iter([0., .25])
    _, _, metrics = train.trial(model, state, cost_weight=2)
    assert metrics["reward"] == metrics["treewidth"] == 0 and metrics["cost_s"] == .25
    assert metrics["selected"] == metrics["useful"] == 0


def test_rejected_contraction_is_finite_negative_and_restores_state(monkeypatch):
    data = _data()
    model = Policy(data["network"], width=8, rounds=1, max_work=1)
    model.reward_colors = ["a"]
    candidate = SummaryClusterCandidate("budget", ("ab", "ac", "bc"), ("a",))
    monkeypatch.setattr(model, "rollout", lambda *a, **k: (candidate, torch.tensor(0.), torch.tensor(0.), []))
    state = train._state(data)
    with train._closed(state, data["samples"][0], model.network["colors"]):
        before = _snapshot(state)
        _, _, metrics = train.trial(model, state)
        assert metrics["rejected"] == 1 and metrics["treewidth"] == 2 and metrics["reward"] == -2
        assert _snapshot(state) == before


def test_training_uses_new_trajectories_and_constant_beta_without_candidate_labels(monkeypatch):
    data, betas = _data(), []
    data["samples"] *= 2
    original = train.trial

    def observed(model, state, **options):
        betas.append(model.beta)
        return original(model, state, **options)

    monkeypatch.setattr(train, "trial", observed)
    model, history = train.fit(data, epochs=3, width=8, rounds=1, batch_size=1, beta=1.5)
    assert betas == [1.5] * 6
    assert model.beta == 1.5 and len(history) == 3
    assert all(row["beta"] == 1.5 for row in history)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert not any("candidates" in sample for sample in data["samples"])


def test_runtime_dispatch_checkpoint_roundtrip_and_single_deserialization(tmp_path, monkeypatch):
    data = _data()
    model = Policy(data["network"], width=8, rounds=1)
    model.reward_colors, model.beta = ["a"], .7
    path = tmp_path / "policy.pt"
    _checkpoint(path, model)
    restored = load(path)
    calls, original = [], torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: calls.append(1) or original(*a, **k))
    dispatched = runtime.load(path)
    assert calls == [1] and dispatched.kind == restored.kind == "construction"
    assert restored.beta == .7 and restored.config == model.config
    assert restored.propose(data["samples"][0]["domains"]) == model.propose(data["samples"][0]["domains"])
    result = train.evaluate(restored, data)
    assert result["states"] == 1 and result["cost_s"] > 0
    assert result["cost_metric"] == train.COST_METRIC and "treewidth" in result
    incompatible = deepcopy(data)
    incompatible["reward_colors"] = ["b"]
    with pytest.raises(ValueError, match="schema"):
        train.evaluate(restored, incompatible)


def test_runtime_removes_summary_and_retries_scalar_contradiction_after_rollback():
    data = _data(odd=True)
    candidate = SummaryClusterCandidate("odd", ("ab", "ac", "bc"), ())
    predictor = SimpleNamespace(kind="construction", network=data["network"],
        config=dict(max_work=1_000_000, max_size=100_000), propose=lambda domains: candidate)
    state = train._state(data)
    for color in "abc":
        state.restrict_domain(color, (0, 1))
    assert state.propagate()
    runtime.attach(state, predictor)
    for _ in range(2):
        state.push_level()
        assert not state.propagate() and state.contradiction
        assert not any(key.startswith("summary_") for key in state._cores)
        state.pop_level()
        assert not state.contradiction
    assert state.on_closure.statistics["selected"] == state.on_closure.statistics["contradictions"] == 2


def _construction_state():
    cores, candidates = {}, []
    for colors in ("abc", "uvw", "xyz"):
        rename = dict(zip("abc", colors))
        keys = []
        for key, core in _network().coresDict.items():
            key = "".join(rename[color] for color in key)
            cores[key] = NumpyCore(values=core.values.copy(), colors=[rename[color] for color in core.colors])
            keys.append(key)
        candidates.append(SummaryClusterCandidate(colors, tuple(keys), (colors[0],)))
    cores.update(ad=NumpyCore(values=np.eye(3, dtype=bool), colors=["a", "d"]),
                 q=NumpyCore(values=np.ones(3, dtype=bool), colors=["q"]))
    network = ConstraintNetwork(cores, maintain_indices=True)
    state = network.create_mac_state({color: (0, 1) for color in "abcuvwxyz"})
    assert state.propagate()
    return network, state, candidates


def _construction_predictor(network, propose):
    return SimpleNamespace(kind="construction", network=network_data(network), propose=propose,
                           config=dict(max_work=1_000_000, max_size=100_000))


def test_ensemble_interleaves_distinct_clusters_and_shares_decoder_budget():
    network, _, (first, second, third) = _construction_state()
    fourth = SummaryClusterCandidate("noop", ("q",), ("q",))
    limits, members = [], {}
    for name, candidates in (("teacher", (first, second)), ("reinforce", (first, third)), ("q", (fourth,))):
        model = _construction_predictor(network, None)
        def proposals(domains, *, max_candidates, max_steps, name=name, candidates=candidates):
            limits.append((name, max_candidates, max_steps))
            return iter(candidates)
        model.proposals = proposals
        members[name] = model
    ensemble = runtime.ConstructionEnsemble(members)
    actual = [(ensemble.last_source, candidate) for candidate in ensemble.proposals({}, max_candidates=10)]
    assert actual == [("teacher", first), ("q", fourth), ("teacher", second), ("reinforce", third)]
    assert limits == [("teacher", 10, 2731), ("reinforce", 10, 2731), ("q", 10, 2730)]
    ensemble = runtime.ConstructionEnsemble(members)
    assert list(ensemble.proposals({}, max_candidates=2)) == [first, fourth]


def test_ensemble_restarts_on_pruning_rotates_members_and_preserves_backtracking(monkeypatch):
    network, state, candidates = _construction_state()
    snapshots, applied, members = [], [], {}
    for name, candidate in zip(("teacher", "reinforce", "q"), candidates):
        model = _construction_predictor(network, None)
        def proposals(domains, *, max_candidates, max_steps, name=name, candidate=candidate):
            snapshots.append((name, dict(domains)))
            yield candidate
            pytest.fail("Pruning must abandon every old construction iterator")
        model.proposals = proposals
        members[name] = model
    ensemble = runtime.ConstructionEnsemble(members)
    apply = runtime.apply_candidate
    def record(state, candidate, prepared):
        applied.append(ensemble.last_source)
        return apply(state, candidate, prepared)
    monkeypatch.setattr(runtime, "apply_candidate", record)
    runtime.attach(state, ensemble, max_candidates=3)
    original, topology = state.snapshot_domains(), tuple(state._cores)
    state.push_level()
    assert state.propagate()
    assert applied == ["teacher", "reinforce", "q"]
    assert snapshots[0][1] == original
    assert snapshots[1][1]["a"] == snapshots[1][1]["d"] == (1,)
    assert snapshots[1][1]["u"] == (0, 1) and snapshots[2][1]["u"] == (1,)
    assert state.domains["x"] == (1,) and state.on_closure.statistics["selected"] == 3
    assert tuple(state._cores) == topology and not state._temporary_core_keys
    state.pop_level()
    assert state.snapshot_domains() == original and state.decision_level == 0


def test_ensemble_rejections_and_duplicates_leave_shared_contraction_slots_available():
    network = ConstraintNetwork(dict(q=NumpyCore(values=np.ones(3), colors=["q"]),
        r=NumpyCore(values=np.ones(3), colors=["r"]),
        xy=NumpyCore(values=np.ones((3, 3)), colors=["x", "y"])))
    state = network.create_mac_state()
    expensive = SummaryClusterCandidate("expensive", ("xy",), ("x",))
    first, second = (SummaryClusterCandidate(color, (color,), (color,)) for color in ("q", "r"))
    members = {}
    for name, candidates in (("teacher", (expensive, first)), ("reinforce", (first, second)), ("q", (first,))):
        model = _construction_predictor(network, None)
        model.config["max_work"] = 3
        model.proposals = lambda domains, candidates=candidates, **kwargs: iter(candidates)
        members[name] = model
    runtime.attach(state, runtime.ConstructionEnsemble(members), max_candidates=2)
    assert state.propagate()
    statistics = state.on_closure.statistics
    assert statistics["selected"] == 2 and statistics["rejected"] == 1
    assert statistics["candidates"] == 3 and statistics["useful"] == 0
    assert not state._temporary_core_keys


@pytest.mark.parametrize("change", ["network", "kind", "max_cores", "max_open_colors", "max_work",
                                    "max_size", "compact_domains", "max_total_work"])
def test_ensemble_requires_compatible_construction_members(change):
    network = _network()
    first, second = (_construction_predictor(network, None) for _ in range(2))
    first.proposals = second.proposals = lambda *args, **kwargs: iter(())
    if change == "network":
        second.network = network_data(_network(odd=True))
    elif change == "kind":
        second.kind = "supervised"
    else:
        second.config[change] = 7
    with pytest.raises(ValueError):
        runtime.ConstructionEnsemble(dict(first=first, second=second))
    with pytest.raises(ValueError, match="at least one member"):
        runtime.ConstructionEnsemble({})


def test_runtime_reselects_after_each_useful_closure_and_stops_at_first_no_information():
    network, state, (first, second, later) = _construction_state()
    noop = SummaryClusterCandidate("noop", ("q",), ("q",))
    proposed, snapshots = iter((first, second, noop, later)), []

    def propose(domains):
        snapshots.append(dict(domains))
        assert not state._queue and not state._queued and not state._temporary_core_keys
        return next(proposed)

    runtime.attach(state, _construction_predictor(network, propose))
    policy, original, topology = state.on_closure, state.snapshot_domains(), tuple(state._cores)
    state.push_level()
    assert state.propagate()
    assert len(snapshots) == 3 and snapshots[0] == original
    assert snapshots[1]["a"] == snapshots[1]["d"] == snapshots[2]["u"] == (1,)
    assert snapshots[1]["u"] == snapshots[2]["x"] == (0, 1)
    assert state.domains["x"] == (0, 1) and not state.is_solved()
    assert policy.statistics["calls"] == 1 and policy.statistics["selected"] == 3
    assert policy.statistics["useful"] == 2 and policy.statistics["removed_values"] == 3
    assert tuple(state._cores) == topology and not state._temporary_core_keys and not state._added_core_keys
    assert state.on_closure is policy and state.decision_level == 1
    assert state.propagate() and len(snapshots) == 3
    state.pop_level()
    assert state.snapshot_domains() == original


def test_runtime_batch_continues_after_noop_and_reselects_on_updated_domains():
    network, state, (first, second, third) = _construction_state()
    noop = SummaryClusterCandidate("noop", ("q",), ("q",))
    snapshots, search_limits = [], []

    def proposals(domains, *, max_candidates):
        snapshots.append(dict(domains))
        search_limits.append(max_candidates)
        assert not state._queue and not state._temporary_core_keys
        if len(domains["a"]) > 1:
            return iter((noop, first))
        if len(domains["u"]) > 1:
            return iter((second,))
        return iter((third,))

    predictor = _construction_predictor(network, lambda domains: pytest.fail("Use construction alternatives"))
    predictor.proposals = proposals
    runtime.attach(state, predictor, max_candidates=4)
    policy, original, topology = state.on_closure, state.snapshot_domains(), tuple(state._cores)
    state.push_level()
    assert state.propagate()
    assert search_limits == [8192] * 3 and len(snapshots) == 3
    assert snapshots[0] == original and snapshots[1]["a"] == snapshots[2]["u"] == (1,)
    assert snapshots[1]["u"] == snapshots[2]["x"] == (0, 1)
    assert state.domains["x"] == (1,)
    assert policy.statistics["candidates"] == policy.statistics["selected"] == 4
    assert policy.statistics["useful"] == 3
    assert tuple(state._cores) == topology and not state._temporary_core_keys
    assert state.propagate() and len(snapshots) == 3
    state.pop_level()
    assert state.snapshot_domains() == original and state.decision_level == 0


@pytest.mark.parametrize("max_candidates,total_work,expected", [(32, None, 32), (32, 3, 1)])
def test_runtime_batch_caps_noop_contractions_and_work(max_candidates, total_work, expected):
    network = ConstraintNetwork({f"q{idx}": NumpyCore(values=np.ones(3), colors=["q"])
                                 for idx in range(40)})
    state = network.create_mac_state()
    assert state.propagate()
    predictor = _construction_predictor(network, lambda domains: pytest.fail("Use construction alternatives"))
    predictor.config["max_total_work"] = total_work
    predictor.proposals = lambda domains, **kwargs: (
        SummaryClusterCandidate(f"q{idx}", (f"q{idx}",), ("q",)) for idx in range(40))
    runtime.attach(state, predictor, max_candidates=max_candidates)
    assert state.propagate() and state.domains["q"] == (0, 1, 2)
    assert state.on_closure.statistics["selected"] == expected
    assert state.on_closure.statistics["useful"] == 0
    if total_work is not None:
        assert state.on_closure.statistics["estimated_work"] == total_work
        assert state.on_closure.statistics["budget_exhausted"] == 1
    assert not state._temporary_core_keys


def test_runtime_batch_rejects_expensive_proposal_then_tries_affordable_one():
    network, state, (expensive, _, _) = _construction_state()
    cheap = SummaryClusterCandidate("noop", ("q",), ("q",))
    predictor = _construction_predictor(network, lambda domains: None)
    predictor.config["max_work"] = 3
    predictor.proposals = lambda domains, **kwargs: iter((expensive, cheap))
    runtime.attach(state, predictor, max_candidates=1)
    before = state.snapshot_domains()
    assert state.propagate() and state.snapshot_domains() == before
    assert state.on_closure.statistics["candidates"] == 2
    assert state.on_closure.statistics["rejected"] == state.on_closure.statistics["selected"] == 1


def test_runtime_batch_legacy_stop_rule_remains_available():
    network, state, _ = _construction_state()
    noop = SummaryClusterCandidate("noop", ("q",), ("q",))
    predictor = _construction_predictor(network, lambda domains: noop)
    predictor.proposals = lambda *a, **k: pytest.fail("Legacy inference uses greedy propose")
    runtime.attach(state, predictor, stop_on_noop=True)
    assert state.propagate() and state.on_closure.statistics["selected"] == 1


def test_runtime_single_proposal_respects_application_allowance():
    network, state, (first, second, later) = _construction_state()
    choices = iter((first, second, later))
    runtime.attach(state, _construction_predictor(network, lambda domains: next(choices)), max_candidates=1)
    before = state.snapshot_domains()
    state.push_level()
    assert state.propagate()
    assert state.on_closure.statistics["selected"] == state.on_closure.statistics["useful"] == 1
    assert state.domains["a"] == (1,) and state.domains["u"] == (0, 1)
    assert not state._temporary_core_keys
    state.pop_level()
    assert state.snapshot_domains() == before


def test_runtime_cached_local_result_stops_reselection_and_replays_after_backtracking(monkeypatch):
    network, state, (candidate, _, _) = _construction_state()
    calls, preparations = [], []
    runtime.attach(state, _construction_predictor(network, lambda domains: calls.append(dict(domains)) or candidate))
    policy, original, prepare = state.on_closure, state.snapshot_domains(), runtime.prepare_candidate
    monkeypatch.setattr(runtime, "prepare_candidate", lambda *a, **k: preparations.append(1) or prepare(*a, **k))
    state.push_level()
    assert state.propagate() and policy.statistics["selected"] == 2 and len(calls) == 2
    prepared = next(iter(policy.plans.values()))
    state.restrict_domain("q", (0, 1))
    assert state.propagate() and policy.statistics["selected"] == 2 and len(calls) == 3
    state.restrict_domain("b", (0,))
    assert state.propagate() and policy.statistics["selected"] == 3 and len(calls) == 4
    state.pop_level()
    assert state.snapshot_domains() == original
    assert state.propagate() and policy.statistics["selected"] == 5 and len(calls) == 6
    assert preparations == [1] and next(iter(policy.plans.values())) is prepared
    assert not state._temporary_core_keys and not any(key.startswith("summary_") for key in state._cores)


def test_runtime_recontracts_successful_cluster_when_new_internal_masks_strengthen_it():
    matrices = [[[0, 1, 0], [1, 0, 1], [0, 1, 1]], [[0, 1, 0], [1, 1, 1], [0, 1, 1]],
                [[1, 1, 1], [1, 0, 1], [1, 0, 0]], [[0, 1, 1], [1, 1, 0], [1, 0, 0]],
                [[1, 1, 1], [0, 0, 1], [1, 1, 1]]]
    network = ConstraintNetwork({key: NumpyCore(values=np.array(values, dtype=bool), colors=list(key))
                                 for key, values in zip(("ab", "ac", "bc", "ad", "bd"), matrices)})
    state, snapshots = network.create_mac_state(), []
    assert state.propagate()
    candidate = SummaryClusterCandidate("triangle", ("ab", "ac", "bc"), ("a",))
    runtime.attach(state, _construction_predictor(network, lambda domains: snapshots.append(dict(domains)) or candidate))
    policy, original = state.on_closure, state.snapshot_domains()
    state.push_level()
    assert state.propagate() and len(snapshots) == 3
    assert snapshots[0]["a"] == snapshots[0]["b"] == (0, 1, 2)
    assert snapshots[1]["a"] == (1, 2) and snapshots[1]["b"] == (0, 2)
    assert snapshots[2]["a"] == state.domains["a"] == (1,) and state.domains["d"] == (0, 1)
    assert policy.statistics["selected"] == 3 and policy.statistics["useful"] == 2
    assert policy.statistics["removed_values"] == 4
    prepared = next(iter(policy.plans.values()))
    state.pop_level()
    assert state.snapshot_domains() == original
    assert state.propagate() and state.domains["a"] == (1,)
    assert policy.statistics["selected"] == 5 and policy.statistics["useful"] == 4
    assert policy.statistics["removed_values"] == 8 and next(iter(policy.plans.values())) is prepared
    assert not state._temporary_core_keys and not any(key.startswith("summary_") for key in state._cores)


def test_runtime_skip_after_useful_cluster_preserves_pruning():
    network, state, (candidate, _, _) = _construction_state()
    candidates = iter((candidate, None))
    runtime.attach(state, _construction_predictor(network, lambda domains: next(candidates)))
    assert state.propagate() and state.domains["a"] == state.domains["d"] == (1,)
    assert state.on_closure.statistics["selected"] == state.on_closure.statistics["useful"] == 1
    assert state.propagate() and not state._temporary_core_keys


def test_runtime_stops_without_proposing_again_when_cluster_solves_network():
    network = ConstraintNetwork({"ab": NumpyCore(values=np.eye(2, dtype=bool), colors=["a", "b"]),
        "bc": NumpyCore(values=np.eye(2, dtype=bool), colors=["b", "c"]),
        "ac": NumpyCore(values=np.array([[1, 1], [1, 0]], dtype=bool), colors=["a", "c"])})
    state, calls = network.create_mac_state(), []
    candidate = SummaryClusterCandidate("solving", ("ab", "bc", "ac"), ("a",))
    runtime.attach(state, _construction_predictor(network, lambda domains: calls.append(1) or candidate))
    assert state.propagate() and state.is_solved() and calls == [1]
    assert state.assignment() == dict(a=0, b=0, c=0)
    assert state.on_closure.statistics["selected"] == state.on_closure.statistics["useful"] == 1
    assert not state._temporary_core_keys


def test_runtime_rejected_best_cluster_stops_before_contraction(monkeypatch):
    network, state, (candidate, _, _) = _construction_state()
    predictor = _construction_predictor(network, lambda domains: candidate)
    predictor.config["max_work"] = 1
    runtime.attach(state, predictor)
    original = state.snapshot_domains()
    monkeypatch.setattr(runtime, "contract_candidate", lambda *a: pytest.fail("Rejected cluster was contracted"))
    assert state.propagate() and state.snapshot_domains() == original
    assert state.on_closure.statistics["rejected"] == state.on_closure.statistics["candidates"] == 1
    assert state.on_closure.statistics["selected"] == 0
    assert state.propagate() and state.on_closure.statistics["calls"] == 1


def test_runtime_later_contraction_error_keeps_earlier_pruning_and_restores_hook(monkeypatch):
    network, state, (first, second, _) = _construction_state()
    candidates, calls = iter((first, second)), []
    runtime.attach(state, _construction_predictor(network, lambda domains: next(candidates)))
    policy, topology, contract = state.on_closure, tuple(state._cores), runtime.contract_candidate

    def failing(candidate, prepared):
        calls.append(candidate)
        if len(calls) == 2:
            raise RuntimeError("injected later contraction failure")
        return contract(candidate, prepared)

    monkeypatch.setattr(runtime, "contract_candidate", failing)
    with pytest.raises(RuntimeError, match="later contraction"):
        state.propagate()
    assert state.domains["a"] == state.domains["d"] == (1,) and state.domains["u"] == (0, 1)
    assert state.on_closure is policy and policy.statistics["selected"] == 1
    assert tuple(state._cores) == topology and not state._temporary_core_keys and state.decision_level == 0
    assert not state._queue and not state._queued


def test_train_and_evaluate_cli_and_refuse_overwrite(tmp_path):
    dataset, checkpoint = tmp_path / "states.json", tmp_path / "new" / "policy.pt"
    dataset.write_text(json.dumps(_data()))
    base = [sys.executable, "-B", "-m", "experiments.cluster_selection.reinforcement.train", str(dataset)]
    command = base + ["--output", str(checkpoint), "--epochs", "2", "--width", "8", "--rounds", "1",
                      "--checkpoint-every", "3", "--lr", ".0001", "--beta", ".7"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "candidate labels are ignored" in result.stdout and checkpoint.exists()
    assert checkpoint.with_name("policy.best.pt").exists()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["cost_metric"] == train.COST_METRIC
    assert saved["config"]["normalize"] and saved["contradiction_reward"] == 1
    assert all(row["cost_metric"] == train.COST_METRIC and "treewidth" in row for row in saved["history"])
    assert all(row["beta"] == .7 for row in saved["history"])
    assert runtime.load(checkpoint).beta == runtime.load(checkpoint.with_name("policy.best.pt")).beta == .7
    result = subprocess.run(base + ["--checkpoint", str(checkpoint)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and '"states": 1' in result.stdout, result.stderr
    before = checkpoint.read_bytes()
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0 and "already exists" in result.stderr and checkpoint.read_bytes() == before


@pytest.mark.parametrize("options", [dict(beta=-1), dict(beta=float("nan")), dict(beta=float("inf")),
                                      dict(cost_weight=-1), dict(lr=0),
                                      dict(heuristic_candidates=-1), dict(heuristic_candidates=.5),
                                      dict(checkpoint_every=0), dict(width=1)])
def test_invalid_training_parameters(options):
    with pytest.raises(ValueError):
        train.fit(_data(), **options)


def test_nonclosed_snapshots_are_rejected_and_reset():
    data = _data()
    state = train._state(data)
    before = _snapshot(state)
    sample = dict(domains=[(0,), (0, 1), (0, 1)])
    with pytest.raises(ValueError, match="GAC-closed"):
        with train._closed(state, sample, data["network"]["colors"]):
            pytest.fail("Nonclosed domains were accepted")
    assert state.snapshot_domains() == before[0] and state.decision_level == 0


def test_normalized_high_degree_encoder_stays_bounded_and_passes_gradients():
    torch.manual_seed(3)
    network = ConstraintNetwork({f"edge_{idx}": NumpyCore(values=np.ones((2, 2)), colors=["hub", f"v{idx}"])
                                 for idx in range(128)}, maintain_indices=True)
    model = Policy(network_data(network), width=8, rounds=6, heuristic_candidates=0)
    with torch.no_grad():
        for module in (model.to_core, model.to_variable, model.update_core, model.update_variable):
            for parameter in module.parameters():
                parameter.mul_(4)
    encoded = model.encode({color: (0, 1) for color in network.dimension_dict})
    for values in encoded:
        assert torch.isfinite(values).all() and values.abs().max() <= 8 ** .5 + 1e-5
        torch.testing.assert_close(values.mean(1), torch.zeros(len(values)), atol=1e-6, rtol=0)
    actions, distribution = model.distribution(encoded, [2] * len(model.network["colors"]), [], [], False, 0)
    (-distribution.log_prob(torch.tensor(actions.index(("core", 0))))).backward()
    for module in (model.variable_encoder, model.core_encoder, model.to_core, model.to_variable, model.head):
        gradient = module[0].weight.grad
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_missing_normalization_flag_preserves_legacy_checkpoint_and_supervised_updates(tmp_path):
    from experiments.cluster_selection.supervised.model import Predictor
    data = _data()
    model = Policy(data["network"], width=8, rounds=3, normalize=False)
    path = tmp_path / "legacy.pt"
    _checkpoint(path, model)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["config"].pop("normalize")
    torch.save(checkpoint, path)
    restored = runtime.load(path)
    assert restored.config["normalize"] is False
    for original, loaded in zip(model.encode(data["samples"][0]["domains"]),
                                restored.encode(data["samples"][0]["domains"])):
        torch.testing.assert_close(original, loaded, atol=0, rtol=0)
    supervised = Predictor(data["network"], width=8, rounds=3)
    variables = supervised.variable_encoder(supervised.domain_features(data["samples"][0]["domains"]))
    cores = supervised.core_encoder(supervised.core_features)
    vi, ci, axes = supervised.variable_idx, supervised.core_idx, supervised.axes
    for _ in range(3):
        messages = supervised.to_core(torch.cat((variables[vi], cores[ci], axes), dim=1))
        cores = cores + supervised.update_core(torch.cat((cores, torch.zeros_like(cores).index_add(0, ci, messages)), dim=1))
        messages = supervised.to_variable(torch.cat((cores[ci], variables[vi], axes), dim=1))
        variables = variables + supervised.update_variable(torch.cat(
            (variables, torch.zeros_like(variables).index_add(0, vi, messages)), dim=1))
    for expected, actual in zip((variables, cores), supervised.encode(data["samples"][0]["domains"])):
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    assert Policy(data["network"], width=1, normalize=False).config["normalize"] is False


@pytest.mark.parametrize("reward", [0., 100_000_000.])
def test_singleton_constant_rewards_have_finite_scaled_advantages_and_nonzero_signal(monkeypatch, reward):
    initial, gradients = [], []
    clip = torch.nn.utils.clip_grad_norm_

    def measured(model, state, **options):
        parameter = next(model.parameters())
        if not initial:
            initial.append(parameter.detach().clone())
        return parameter.flatten()[0], parameter.sum() * 0, dict(reward=reward, gain=0, treewidth=0,
            cost_s=0., selected=1, useful=0, contradictions=0, rejected=0)

    def clipped(parameters, maximum, **options):
        parameters = list(parameters)
        gradients.append(float(torch.stack([p.grad.norm() for p in parameters if p.grad is not None]).norm()))
        return clip(parameters, maximum, **options)

    monkeypatch.setattr(train, "trial", measured)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clipped)
    model, _ = train.fit(_data(), epochs=3, batch_size=1, width=8, rounds=1, entropy_weight=0)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert all(np.isfinite(value) for value in gradients)
    if reward:
        assert gradients[0] > 1_000_000 and 0 < gradients[-1] < 100
        assert not torch.equal(initial[0], next(model.parameters()))
    else:
        assert gradients == [0., 0., 0.]


def test_checkpoint_selection_uses_post_update_weights_and_common_inference_beta(tmp_path, monkeypatch):
    snapshots, scores = [], iter((1., 3., 2.))

    def evaluate(model, data, **options):
        assert model.beta == .5 and options["state"] is not None
        snapshots.append({key: value.detach().clone() for key, value in model.state_dict().items()})
        return dict(reward=next(scores), useful=1., states=1, cost_metric=train.COST_METRIC)

    monkeypatch.setattr(train, "evaluate", evaluate)
    output = tmp_path / "new" / "policy.pt"
    model, history = train.fit(_data(), epochs=3, width=8, rounds=1, beta=.5,
                              output=output, checkpoint_every=1, cost_weight=.2, contradiction_reward=2)
    latest = torch.load(output, weights_only=True)
    best = torch.load(output.with_name("policy.best.pt"), weights_only=True)
    assert latest["history"][-1]["epoch"] == 3 and best["history"][-1]["epoch"] == 2
    assert best["selection"]["source"] == "training" and best["selection"]["reward"] == 3.
    assert best["beta"] == best["selection"]["beta"] == latest["beta"] == model.beta == .5
    assert latest["cost_weight"] == best["cost_weight"] == .2
    assert latest["contradiction_reward"] == best["contradiction_reward"] == 2
    assert len(history) == len(snapshots) == 3
    for key in model.state_dict():
        torch.testing.assert_close(latest["weights"][key], snapshots[-1][key], atol=0, rtol=0)
        torch.testing.assert_close(best["weights"][key], snapshots[1][key], atol=0, rtol=0)


@pytest.mark.parametrize("existing", ["policy.pt", "policy.best.pt"])
def test_checkpoint_preflight_protects_both_artifacts_before_training(tmp_path, monkeypatch, existing):
    output, occupied = tmp_path / "policy.pt", tmp_path / existing
    occupied.write_bytes(b"preserve")
    monkeypatch.setattr(train, "_state", lambda data: pytest.fail("Training began before path validation"))
    with pytest.raises((ValueError, FileExistsError), match="exists"):
        train.fit(_data(), output=output)
    assert occupied.read_bytes() == b"preserve"
    assert sorted(path.name for path in tmp_path.iterdir()) == [existing]


def test_interrupted_training_keeps_completed_checkpoint_with_constant_beta(tmp_path, monkeypatch):
    original, calls = train.trial, []

    def interrupted(*args, **options):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return original(*args, **options)

    monkeypatch.setattr(train, "trial", interrupted)
    monkeypatch.setattr(train, "evaluate", lambda *a, **k: dict(reward=1., useful=1., states=1))
    output = tmp_path / "policy.pt"
    with pytest.raises(KeyboardInterrupt):
        train.fit(_data(), epochs=3, width=8, rounds=1, beta=1.5, output=output, checkpoint_every=1)
    latest = torch.load(output, weights_only=True)
    best = torch.load(output.with_name("policy.best.pt"), weights_only=True)
    assert latest["history"][-1]["epoch"] == best["history"][-1]["epoch"] == 1
    assert latest["beta"] == best["beta"] == 1.5
    assert runtime.load(output).config["normalize"] and runtime.load(output.with_name("policy.best.pt")).beta == 1.5


@pytest.mark.parametrize("mode", ["plain", "trained", "first", "heuristic"])
def test_experiment_validates_complete_generic_dfs_solutions_and_cleans_state(monkeypatch, mode):
    from experiments.cluster_selection.reinforcement import experiment
    network = _network()
    domains = {color: (0, 1) for color in "abc"}
    model = Policy(network_data(network), width=8, rounds=1, max_cores=3)
    model.reward_colors = ["a"]
    states, create = [], network.create_mac_state
    monkeypatch.setattr(network, "create_mac_state", lambda values: states.append(create(values)) or states[-1])
    row = experiment.solve(network, domains, mode, experiment.First(model) if mode == "first" else model,
                           nodes=20, seconds=10)
    assert row["status"] == "solved" and row["solved"] and row["valid"]
    assert set(row["assignment"]) == set("abc")
    assert all(value in domains[color] for color, value in row["assignment"].items())
    assert all(core[row["assignment"]] != 0 for core in network.coresDict.values())
    assert row["work"]["max_tensor_size"] <= model.config["max_size"]
    assert (row["work"]["contractions"] > 0) == (mode != "plain")
    assert states[0].decision_level == 0 and states[0]._cores.keys() == network.coresDict.keys()
    assert not states[0]._temporary_core_keys
    limited = experiment.solve(network, domains, "plain", model, nodes=1, seconds=10)
    assert limited["status"] == "node_limit" and limited["nodes"] == 1 and not limited["valid"]


def test_experiment_cumulative_work_budget_retains_pruning_and_cleans_temporary_summary(monkeypatch):
    from experiments.cluster_selection.reinforcement import experiment
    from experiments.cluster_selection.candidate_evaluation_mac import prepare_candidate
    network, domains = _network(), {color: (0, 1) for color in "abc"}
    candidate = SummaryClusterCandidate("budget", ("ab", "ac", "bc"), ("a",))
    model = _construction_predictor(network, lambda values: candidate)
    state = network.create_mac_state(domains)
    assert state.propagate()
    budget = prepare_candidate(state, candidate)[0].work
    monkeypatch.setattr(network, "create_mac_state", lambda values: state)
    row = experiment.solve(network, domains, "trained", model, nodes=20, seconds=10, total_work=budget)
    assert row["status"] == "solved" and row["solved"] and row["valid"]
    assert row["work"]["budget_exhausted"]
    assert row["work"]["contractions"] == 1 and row["work"]["estimated_work"] == budget
    assert state.domains["a"] == (1,) and row["cluster_statistics"]["useful"] == 1
    assert state.decision_level == 0 and not state._temporary_core_keys
    assert state._cores.keys() == network.coresDict.keys()


def test_experiment_timeout_restores_contraction_function_and_alarm(monkeypatch):
    import signal
    from experiments.cluster_selection.reinforcement import experiment
    network = _network()
    original, handler = runtime.contract_candidate, signal.getsignal(signal.SIGALRM)

    def timed_out(**options):
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
        raise TimeoutError("injected timeout")

    monkeypatch.setattr(network, "solve_mac", timed_out)
    row = experiment.solve(network, {}, "plain", None, nodes=20, seconds=10)
    assert row["status"] == "wall_timeout" and not row["valid"]
    assert runtime.contract_candidate is original and signal.getsignal(signal.SIGALRM) is handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


@pytest.mark.parametrize("assignment", [dict(a=1), dict(a=0, b=0, c=0), dict(a=2, b=0, c=1)])
def test_experiment_rejects_partial_or_invalid_solver_assignments(monkeypatch, assignment):
    from experiments.cluster_selection.reinforcement import experiment
    network = _network()
    monkeypatch.setattr(network, "solve_mac", lambda **options: SimpleNamespace(
        solved=True, assignment=assignment, nodes=1, node_limit_reached=False, unsatisfiable=False))
    row = experiment.solve(network, {color: (0, 1) for color in "abc"}, "plain", None, nodes=20, seconds=10)
    assert row["status"] == "invalid" and not row["solved"] and not row["valid"]


@pytest.mark.parametrize("decoder", ["depth_first", "discrepancy"])
def test_experiment_cli_exclusive_output_and_overlap_uses_domains_instead_of_ids(tmp_path, decoder):
    raw = network_data(_network())
    model = Policy(raw, width=8, rounds=1, max_cores=3)
    model.reward_colors = ["a"]
    checkpoint, evaluation, training = (tmp_path / name for name in ("policy.pt", "eval.json", "train.json"))
    _checkpoint(checkpoint, model)
    data = dict(network=raw, initial_domains=[{color: [0, 1] for color in "abc"}],
                instance_ids=["evaluation"], reward_colors=["a"])
    evaluation.write_text(json.dumps(data))
    training.write_text(json.dumps({**data, "initial_domains": [dict(a=[1], b=[0, 1], c=[0, 1])],
                                    "instance_ids": ["training"]}))
    output = tmp_path / "solves.jsonl"
    command = [sys.executable, "-B", "-m", "experiments.cluster_selection.reinforcement.experiment",
               str(evaluation), "--training-data", str(training), "--checkpoint", str(checkpoint),
               "--output", str(output), "--indices", "0", "--nodes", "20", "--seconds", "10",
               "--total-work", "100000", "--decoder", decoder]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]["exact_training_overlap"] == 0 and rows[0]["threads"] == 1
    assert rows[0]["total_work_budget"] == 100000
    assert rows[0]["config"]["decoder"] == decoder
    assert {row["mode"] for row in rows[1:]} == {"plain", "trained", "untrained", "prior", "first", "heuristic"}
    assert all(row["status"] == "solved" and row["valid"] and row["targets"]["a"] == 1 for row in rows[1:])
    assert all(row["work"]["estimated_work"] <= 100000 for row in rows[1:])
    saved = output.read_bytes()
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0 and output.read_bytes() == saved
    training.write_text(json.dumps({**data, "initial_domains": [{color: [1, 0, 1] for color in "cba"}],
                                    "instance_ids": ["different-id"]}))
    command[command.index(str(output))] = str(tmp_path / "overlap.jsonl")
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0 and "overlap" in result.stderr.lower()
    assert not (tmp_path / "overlap.jsonl").exists()
