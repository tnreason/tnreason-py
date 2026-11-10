from copy import deepcopy
from types import SimpleNamespace
import subprocess
import sys
import numpy as np
import pytest
from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.cluster_selection.data import network_data
from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from experiments.cluster_selection.candidate_evaluation_mac import evaluate_candidate, prepare_candidate
from experiments.cluster_selection import runtime


def _network(extra=False):
    relation = np.array([[1, 0, 1], [1, 1, 1], [1, 1, 1]])
    cores = {"ab": NumpyCore(values=relation, colors=["a", "b"]),
             "ac": NumpyCore(values=relation, colors=["a", "c"]),
             "bc": NumpyCore(values=~np.eye(3, dtype=bool), colors=["b", "c"]),
             "ad": NumpyCore(values=np.eye(3, dtype=bool), colors=["a", "d"])}
    if extra:
        cores["e"] = NumpyCore(values=np.ones(3), colors=["e"])
    return ConstraintNetwork(cores, maintain_indices=True)


def _state(network=None):
    network = _network() if network is None else network
    state = network.create_mac_state({"a": (0, 1), "b": (0, 1), "c": (0, 1)})
    assert state.propagate()
    return state


def _candidate():
    return SummaryClusterCandidate("triangle", ("ab", "ac", "bc"), ("a",))


def _topology(state):
    return (tuple(state._cores), {key: id(value) for key, value in state._contractions.items()},
            dict(state._core_keys_by_color), tuple(state._temporary_core_keys), frozenset(state._added_core_keys),
            tuple(state._queue), frozenset(state._queued), state.decision_level)


def _predictor(network, choice=0):
    return SimpleNamespace(network=network_data(network), select=lambda *args, **kwargs: choice)


def test_transient_summary_keeps_pruning_counters_and_normal_branch_rollback():
    state = _state()
    candidate = _candidate()
    expected = evaluate_candidate(state, candidate)
    root = state.snapshot_domains()
    state.push_level()
    state._added_core_keys.add("summary_triangle")
    topology, before = _topology(state), state.statistics
    domain_tensors = {color: id(value) for color, value in state._domain_tensors.items()}
    state.on_closure = lambda *args: pytest.fail("Recursive cluster selection")
    hook = state.on_closure
    result = runtime.apply_candidate(state, candidate)
    assert state.snapshot_domains() == expected.domains
    assert state.domains["a"] == state.domains["d"] == (1,)
    assert result.removed_values == 2
    assert state.statistics.removed_values == before.removed_values + 2
    assert _topology(state) == topology and state.on_closure is hook
    assert {color: id(value) for color, value in state._domain_tensors.items()} == domain_tensors
    state.pop_level()
    assert state.snapshot_domains() == root and state.decision_level == 0


@pytest.mark.parametrize("stage", ["contraction", "propagation"])
def test_application_errors_restore_domains_queue_hook_and_topology(monkeypatch, stage):
    state, candidate = _state(), _candidate()
    state.on_closure = lambda *args: None
    topology, domains, trail, hook = _topology(state), state.snapshot_domains(), tuple(state._trail), state.on_closure

    def fail(*args, **kwargs):
        if stage == "propagation":
            state.restrict_domain("a", (1,))
        raise RuntimeError("injected failure")

    monkeypatch.setattr(runtime if stage == "contraction" else state,
                        "contract_candidate" if stage == "contraction" else "propagate", fail)
    with pytest.raises(RuntimeError, match="injected"):
        runtime.apply_candidate(state, candidate)
    assert state.snapshot_domains() == domains and tuple(state._trail) == trail
    assert _topology(state) == topology and state.on_closure is hook


def test_scalar_contradiction_survives_cleanup_but_not_parent_pop():
    cores = {key: NumpyCore(values=~np.eye(3, dtype=bool), colors=list(key)) for key in ("ab", "bc", "ca")}
    state = ConstraintNetwork(cores).create_mac_state({color: (0, 1) for color in "abc"})
    assert state.propagate()
    topology = _topology(state)
    state.push_level()
    result = runtime.apply_candidate(state, SummaryClusterCandidate("odd", tuple(cores), ()))
    assert result.contradiction and state.contradiction
    assert not state._queue and not state._queued and not state._temporary_core_keys
    assert tuple(state._cores) == topology[0]
    state.pop_level()
    assert not state.contradiction and _topology(state) == topology


def test_policy_prepares_matching_features_and_contracts_only_selected_candidate(monkeypatch):
    network = _network()
    state = _state(network)
    candidate = _candidate()
    expected = evaluate_candidate(state, candidate)
    calls, contractions = [], []
    predictor = _predictor(network)

    def select(domains, candidates, **kwargs):
        calls.append(candidates)
        assert candidates[0]["estimate"] == [expected.estimate.work, expected.estimate.max_size, expected.estimate.output_size]
        assert not any(key in candidates[0] for key in ("gain", "time", "contradiction"))
        return 0

    predictor.select = select
    runtime.attach(state, predictor)
    policy = state.on_closure
    monkeypatch.setattr(policy.generator, "generate", lambda *args, **kwargs: (candidate,))
    original = runtime.contract_candidate
    monkeypatch.setattr(runtime, "contract_candidate", lambda *args: contractions.append(1) or original(*args))
    assert state.propagate()
    assert state.snapshot_domains() == expected.domains and len(calls) == len(contractions) == 1
    assert state.propagate() and len(calls) == 1
    assert policy.statistics["selected"] == 1 and policy.statistics["removed_values"] == 2
    assert not any(key.startswith("summary_") for key in state._cores)


def test_skip_policy_matches_plain_search_and_default_imports_stay_torch_free():
    network = _network()
    plain = network.solve_mac({"a": (0, 1), "b": (0, 1), "c": (0, 1)})
    state = runtime.attach(network.create_mac_state({"a": (0, 1), "b": (0, 1), "c": (0, 1)}),
                           _predictor(network, choice=None))
    learned = network.solve_mac(state=state)
    assert learned == plain
    assert state.on_closure.statistics["selected"] == 0
    result = subprocess.run([sys.executable, "-B", "-c",
        "import sys; import experiments.constraint_networks.mac; import experiments.cluster_selection.runtime; "
        "assert 'torch' not in sys.modules"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_no_recontraction_for_unrelated_changes_but_reconsider_after_local_changes(monkeypatch):
    network = _network(extra=True)
    state = runtime.attach(_state(network), _predictor(network))
    policy, candidate = state.on_closure, _candidate()
    monkeypatch.setattr(policy.generator, "generate", lambda *args, **kwargs: (candidate,))
    assert state.propagate() and policy.statistics["selected"] == 1
    state.push_level()
    state.restrict_domain("e", (0, 1))
    assert state.propagate() and policy.statistics["selected"] == 1
    state.assign("b", 1)
    assert state.propagate() and policy.statistics["selected"] == 2
    state.pop_level()
    assert state.propagate() and policy.statistics["selected"] == 3
    assert len(policy.plans) == 1


def test_schema_matching_ignores_order_and_allows_only_unary_extra_evidence():
    network = _network()
    predictor = _predictor(network)
    reordered = ConstraintNetwork(dict(reversed(tuple(network.coresDict.items()))), maintain_indices=True)
    runtime.attach(_state(reordered), predictor)
    evidence = ConstraintNetwork({**network.coresDict, "given": NumpyCore(values=np.ones(3), colors=["a"])},
                                 maintain_indices=True)
    runtime.attach(_state(evidence), predictor)
    bad = deepcopy(network.coresDict)
    bad["ab"].values[0, 0] = 0
    with pytest.raises(ValueError, match="schema"):
        runtime.attach(_state(ConstraintNetwork(bad)), predictor)
    extra = ConstraintNetwork({**network.coresDict, "other": NumpyCore(values=np.ones((3, 3)), colors=["a", "b"])})
    with pytest.raises(ValueError, match="schema"):
        runtime.attach(_state(extra), predictor)
    state = _state()
    state.on_closure = lambda *args: None
    with pytest.raises(ValueError, match="another"):
        runtime.attach(state, predictor)


def test_checkpoint_prediction_can_be_used_during_actual_search(tmp_path):
    torch = pytest.importorskip("torch")
    from experiments.cluster_selection.supervised.model import Predictor
    network = _network()
    model = Predictor(network_data(network), width=8, rounds=1)
    with torch.no_grad():
        for parameter in model.head.parameters():
            parameter.zero_()
        model.head[-1].bias[0] = 1
    path = tmp_path / "predictor.pt"
    torch.save(dict(version=1, network=model.network, reward_colors=list(network.dimension_dict),
                    config=model.config, weights=model.state_dict()), path)
    state = runtime.attach(network.create_mac_state({"a": (0, 1), "b": (0, 1), "c": (0, 1)}), path, max_candidates=4)
    result = network.solve_mac(state=state)
    assert result.solved and not result.node_limit_reached
    assert state.on_closure.statistics["selected"] > 0
    assert all(core[{color: result.assignment[color] for color in core.colors}] != 0
               for core in network.coresDict.values())
    assert not state._temporary_core_keys and not any(key.startswith("summary_") for key in state._cores)


def test_queued_states_and_budget_limits_reject_before_contraction(monkeypatch):
    state = _state()
    before = _topology(state), state.snapshot_domains()
    monkeypatch.setattr(runtime, "contract_candidate", lambda *args: pytest.fail("Contraction must not run"))
    with pytest.raises(ValueError, match="budget"):
        runtime.apply_candidate(state, _candidate(), max_work=1)
    assert (_topology(state), state.snapshot_domains()) == before
    state.assign("b", 0)
    with pytest.raises(ValueError, match="closure"):
        runtime.apply_candidate(state, _candidate())


def test_heuristic_applies_the_whole_batch_and_keeps_reversible_pruning(monkeypatch):
    cores = _network().coresDict.copy()
    rename = dict(zip("abcd", "uvwx"))
    cores.update({"other_" + key: NumpyCore(values=core.values.copy(), colors=[rename[c] for c in core.colors])
                  for key, core in _network().coresDict.items()})
    state = ConstraintNetwork(cores).create_mac_state({c: (0, 1) for c in "abcuvw"})
    assert state.propagate()
    before = state.snapshot_domains()
    runtime.attach_heuristic(state)
    policy, calls = state.on_closure, []
    candidates = (_candidate(), SummaryClusterCandidate("other", ("other_ab", "other_ac", "other_bc"), ("u",)))

    def generate(*args, **kwargs):
        assert kwargs["cyclic_only"] is True
        calls.append(1)
        return candidates

    monkeypatch.setattr(policy.generator, "generate", generate)
    state.push_level()
    assert state.propagate()
    assert state.domains["a"] == state.domains["u"] == (1,)
    assert policy.statistics["proposed"] == policy.statistics["applied"] == policy.statistics["useful"] == 2
    assert policy.statistics["removed_values"] == 4 and len(calls) == 1
    assert state.on_closure is policy and not state._temporary_core_keys
    assert not any(key.startswith("summary_") for key in state._cores)
    assert state.propagate() and len(calls) == 1
    state.pop_level()
    assert state.snapshot_domains() == before
    assert state.propagate() and policy.statistics["applied"] == 4


def test_heuristic_noop_cache_uses_internal_domains_and_survives_backtracking(monkeypatch):
    cores = {key: NumpyCore(values=np.ones((3, 3), dtype=bool), colors=list(key)) for key in ("ab", "bc", "ca")}
    cores["e"] = NumpyCore(values=np.ones(3), colors=["e"])
    state = ConstraintNetwork(cores).create_mac_state()
    assert state.propagate()
    runtime.attach_heuristic(state)
    policy, candidate = state.on_closure, SummaryClusterCandidate("noop", ("ab", "bc", "ca"), ("a",))
    monkeypatch.setattr(policy.generator, "generate", lambda *args, **kwargs: (candidate,))
    assert state.propagate()
    prepared = next(iter(policy.plans.values()))
    assert policy.statistics["applied"] == 1 and policy.statistics["useful"] == 0
    state.push_level()
    state.restrict_domain("e", (0, 1))
    assert state.propagate() and policy.statistics["applied"] == 1 and policy.statistics["skipped"] == 1
    state.restrict_domain("b", (0, 1))
    assert state.propagate() and policy.statistics["applied"] == 2
    assert next(iter(policy.plans.values())) is prepared
    state.pop_level()
    assert state.propagate() and policy.statistics["applied"] == 3
    assert state.propagate() and policy.statistics["applied"] == 3


def test_heuristic_stops_on_contradiction_and_can_reconsider_after_pop(monkeypatch):
    cores = {key: NumpyCore(values=~np.eye(3, dtype=bool), colors=list(key)) for key in ("ab", "bc", "ca")}
    state = ConstraintNetwork(cores).create_mac_state({color: (0, 1) for color in "abc"})
    assert state.propagate()
    runtime.attach_heuristic(state)
    policy = state.on_closure
    candidate = SummaryClusterCandidate("odd", tuple(cores), ())
    monkeypatch.setattr(policy.generator, "generate", lambda *args, **kwargs: (candidate, candidate))
    state.push_level()
    assert not state.propagate() and policy.statistics["applied"] == policy.statistics["useful"] == 1
    assert policy.statistics["contradictions"] == 1 and not state._temporary_core_keys
    state.pop_level()
    assert not state.contradiction
    state.push_level()
    assert not state.propagate() and policy.statistics["applied"] == 2
    state.pop_level()


def test_heuristic_rejects_over_budget_plans_and_existing_hooks(monkeypatch):
    state = _state()
    runtime.attach_heuristic(state, max_work=1)
    policy = state.on_closure
    monkeypatch.setattr(policy.generator, "generate", lambda *args, **kwargs: (_candidate(),))
    assert state.propagate() and policy.statistics["rejected"] == 1 and policy.statistics["applied"] == 0
    with pytest.raises(ValueError, match="another"):
        runtime.attach_heuristic(state)
    with pytest.raises(ValueError, match="bound"):
        policy(_state(), set())
