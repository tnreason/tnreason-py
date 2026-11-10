from copy import deepcopy
from hashlib import sha256
from math import ceil, log
from random import Random
from types import SimpleNamespace
import numpy as np
import pytest
from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.cluster_selection import runtime
from experiments.cluster_selection.candidate_evaluation_mac import evaluate_candidate
from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from experiments.cluster_selection.data import network_data

torch = pytest.importorskip("torch")
from experiments.cluster_selection.reinforcement import learn
from experiments.cluster_selection.reinforcement.model import Policy, load
from experiments.cluster_selection.reinforcement.train import _closed


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _data():
    """Two disconnected generic CSPs, each forced only by a triangle contraction."""
    closing = np.ones((4, 4), dtype=bool)
    closing[1, 1] = closing[3, 3] = False
    cores = {}
    for a, b, c in ("abc", "uvw"):
        cores[a + b] = NumpyCore(values=np.eye(4, dtype=bool), colors=[a, b])
        cores[b + c] = NumpyCore(values=np.eye(4, dtype=bool), colors=[b, c])
        cores[a + c] = NumpyCore(values=closing.copy(), colors=[a, c])
    network = ConstraintNetwork(cores, maintain_indices=True)
    raw = network_data(network)
    initial = [{color: list(values) for color in raw["colors"]} for values in ((0, 1), (2, 3))]
    samples = [dict(puzzle=idx, domains=[domains[color] for color in raw["colors"]])
               for idx, domains in enumerate(initial)]
    return dict(version=1, network=raw, reward_colors=raw["colors"], initial_domains=initial, samples=samples)


def _model(data):
    return SimpleNamespace(config=dict(max_work=10_000, max_size=256, compact_domains=True),
                           reward_colors=data["reward_colors"])


def _candidate(outputs=("a",)):
    return SummaryClusterCandidate("triangle", ("ab", "bc", "ac"), outputs)


def _snapshot(state):
    return (state.snapshot_domains(), tuple(state._cores), tuple(state._trail), state.decision_level,
            tuple(state._queue), frozenset(state._queued), frozenset(state._added_core_keys))


def _causal_snapshot(state):
    return (_snapshot(state), state.statistics, state.on_closure,
            tuple((key, id(value)) for key, value in vars(state).items()),
            tuple(state._temporary_core_keys), frozenset(state._changed_colors), tuple(state._level_markers),
            state.contradiction, state.contradiction_reason,
            tuple((key, core.values.tobytes()) for key, core in state._cores.items()),
            tuple((color, values.tobytes()) for color, values in state._domain_tensors.items()))


def test_causal_witness_replays_failed_assignment_and_prunes_without_touching_source_hook_or_snapshot():
    data = _data()
    _, state, pools, _ = learn.training_states(data, [0], [1])
    with _closed(state, pools[0][0], data["network"]["colors"]):
        state.on_closure = lambda *args: pytest.fail("Probes must never call the source hook")
        before = _causal_snapshot(state)
        candidates = learn.causal_candidates(state, ["a"], Random(13), 2, max_cores=3)
        assert len(candidates) == 1 and candidates[0].open_colors == ("a",)
        assert set(candidates[0].core_keys) == {"ab", "bc", "ac"}
        assert _causal_snapshot(state) == before
        witness = ConstraintNetwork({key: state._cores[key] for key in candidates[0].core_keys}, maintain_indices=True)
        for value in (0, 1):
            replay = witness.create_mac_state({color: state.domains[color] for color in witness.dimension_dict})
            assert replay.propagate()
            replay.assign("a", value)
            assert bool(replay.propagate()) == (value == 0)
        result = evaluate_candidate(state, candidates[0], compact_domains=True)
        assert result.removed_domain_values == 3 and result.domains["a"] == (0,)
        assert _snapshot(state) == before[0] and state.statistics == before[1] and state.on_closure is before[2]
        before = _causal_snapshot(state)
        # MAC snapshots remain authoritative after their mutable parent changes.
        state.constraint_network.coresDict["ac"].values[:] = True
        assert learn.causal_candidates(state, ["a"], Random(13), 2, max_cores=3) == candidates
        assert _causal_snapshot(state) == before
        state.on_closure = None


def test_causal_witness_includes_branch_added_constraint_from_the_current_state():
    network = ConstraintNetwork({key: NumpyCore(values=values, colors=list(key)) for key, values in (
        ("ab", np.eye(2, dtype=bool)), ("bc", np.eye(2, dtype=bool)),
        ("ac", np.ones((2, 2), dtype=bool)))}, maintain_indices=True)
    state = network.create_mac_state()
    assert state.propagate()
    assert not learn.causal_candidates(state, ["a"], Random(2), 2, max_cores=3)
    state.push_level()
    state.add_cores({"tight": NumpyCore(values=np.array([[1, 1], [1, 0]], dtype=bool), colors=["a", "c"])},
                    reversible=True)
    assert state.propagate()
    before = _causal_snapshot(state)
    candidates = learn.causal_candidates(state, ["a"], Random(2), 2, max_cores=3)
    assert len(candidates) == 1 and "tight" in candidates[0].core_keys
    assert _causal_snapshot(state) == before
    assert evaluate_candidate(state, candidates[0], compact_domains=True).removed_domain_values == 3
    assert _snapshot(state) == before[0] and state.statistics == before[1]
    state.pop_level()
    assert "tight" not in state._cores


def test_causal_probes_require_closed_domains_and_respect_zero_no_conflict_and_core_limits():
    data = _data()
    _, state, pools, _ = learn.training_states(data, [0], [1])
    with _closed(state, pools[0][0], data["network"]["colors"]):
        before = _causal_snapshot(state)
        assert not learn.causal_candidates(state, ["a"], Random(1), 0, max_cores=3)
        assert not learn.causal_candidates(state, ["a"], Random(1), 2, max_cores=2)
        for limit, max_cores in ((-1, 3), (2, 0)):
            with pytest.raises(ValueError, match="nonnegative probes"):
                learn.causal_candidates(state, ["a"], Random(1), limit, max_cores=max_cores)
        assert _causal_snapshot(state) == before
    easy = ConstraintNetwork({"ab": NumpyCore(values=np.ones((2, 2), dtype=bool), colors=["a", "b"])},
                             maintain_indices=True).create_mac_state()
    assert easy.propagate()
    assert not learn.causal_candidates(easy, ["a", "b"], Random(1), 4, max_cores=1)
    pending = ConstraintNetwork({"a": NumpyCore(values=np.array([1, 0], dtype=bool), colors=["a"])},
                                maintain_indices=True).create_mac_state()
    before = _causal_snapshot(pending)
    with pytest.raises(ValueError, match="GAC-closed"):
        learn.causal_candidates(pending, ["a"], Random(1), 2, max_cores=1)
    assert _causal_snapshot(pending) == before


def test_causal_probe_exception_restores_scratch_domains_levels_and_temporary_method(monkeypatch):
    data = _data()
    _, state, pools, _ = learn.training_states(data, [0], [1])
    create, scratch_states = ConstraintNetwork.create_mac_state, []

    def failing_probe(network, domains=None):
        scratch = create(network, domains)
        assign = scratch.assign

        def fail_after_assignment(color, value):
            assign(color, value)
            raise RuntimeError("probe failed")

        scratch.assign = fail_after_assignment
        scratch_states.append(scratch)
        return scratch

    with _closed(state, pools[0][0], data["network"]["colors"]):
        before = _causal_snapshot(state)
        monkeypatch.setattr(ConstraintNetwork, "create_mac_state", failing_probe)
        with pytest.raises(RuntimeError, match="probe failed"):
            learn.causal_candidates(state, ["a"], Random(3), 2, max_cores=3)
        assert _causal_snapshot(state) == before and len(scratch_states) == 1
        scratch = scratch_states[0]
        assert scratch.snapshot_domains() == state.snapshot_domains() and scratch.decision_level == 0
        assert not scratch._trail and not scratch._queue and not scratch.contradiction
        assert "_set_domain" not in vars(scratch) and scratch.on_closure is None


def test_causal_teacher_fit_records_probes_without_calling_them_structural_or_unseen(tmp_path, monkeypatch):
    monkeypatch.setattr(learn.ClusterCandidateGenerator, "generate", lambda *a, **k: ())
    output = tmp_path / "causal.pt"
    model, history = learn.fit(_data(), output=output, epochs=1, width=8, rounds=1, seed=7,
        rollouts=2, steps=1, warmup=1, teacher_candidates=8, teacher_mutations=0, teacher_probes=12,
        train_indices=[0], validation_indices=[1], validation_seconds=2., max_cores=3,
        max_open_colors=1, max_work=10_000, max_size=256, target_first=True, value_encoding=True)
    row = history[0]
    assert row["teacher_probe_assumptions"] == 12 and row["teacher_probe_candidates"] == row["teacher_trials"] == 6
    assert row["teacher_probe_seconds"] >= 0
    assert row["useful"] == row["useful_novel"] == row["novel"] == row["trials"] == 2
    assert row["useful_unseen"] == 0  # Causal teachers were actually measured on this state.
    checkpoint = torch.load(output, weights_only=True)
    assert checkpoint["training"]["teacher_probes"] == 12
    assert runtime.load(output).config == model.config


def test_measure_rewards_actual_pruning_and_penalizes_completed_noops_and_skips():
    data = _data()
    _, state, pools, _ = learn.training_states(data, [0], [1])
    original = _snapshot(state)
    with _closed(state, pools[0][0], data["network"]["colors"]):
        before = _snapshot(state)
        useful = learn.measure(_model(data), state, _candidate(), known_satisfiable=True)
        noop = learn.measure(_model(data), state, SummaryClusterCandidate("edge", ("ab",), ("a",)))
        skip = learn.measure(_model(data), state, None)
        assert useful["useful"] == 1 and useful["gain"] == 3 and useful["reward"] > 0
        assert useful["information"] > 0 and not useful["solved"]
        assert noop["useful"] == noop["gain"] == 0 and noop["reward"] < 0
        assert skip["selected"] == 0 and skip["reward"] < 0
        assert _snapshot(state) == before
    assert _snapshot(state) == original


def test_contradiction_reward_has_no_wipeout_gain_and_guards_known_satisfiable_states():
    cores = {key: NumpyCore(values=~np.eye(3, dtype=bool), colors=list(key)) for key in ("ab", "bc", "ac")}
    state = ConstraintNetwork(cores).create_mac_state({color: (0, 1) for color in "abc"})
    assert state.propagate()
    model = SimpleNamespace(config=dict(max_work=10_000, max_size=256, compact_domains=True),
                            reward_colors=list("abc"))
    before = _snapshot(state)
    for outputs in ((), ("a",), ("c", "b", "a")):
        candidate = _candidate(outputs)
        result = learn.measure(model, state, candidate, contradiction_reward=2., cost_weight=0.)
        assert result["contradictions"] == result["useful"] == 1
        assert result["gain"] == result["information"] == result["solved"] == 0
        assert result["reward"] == pytest.approx(2. - .001 - .01 * result["cost_s"])
        assert _snapshot(state) == before
        with pytest.raises(AssertionError, match="known satisfiable"):
            learn.measure(model, state, candidate, known_satisfiable=True)
        assert _snapshot(state) == before


def test_already_solved_input_cannot_reward_a_completed_noop():
    data = _data()
    _, state, _, _ = learn.training_states(data, [0], [1])
    for color in state.domains:
        state.assign(color, 0)
    assert state.propagate() and state.is_solved()
    result = learn.measure(_model(data), state, _candidate(), known_satisfiable=True)
    assert result["gain"] == result["useful"] == 0
    assert result["reward"] < 0


def test_remaining_work_budget_rejects_a_cluster_before_trial_mutation():
    data = _data()
    _, state, pools, _ = learn.training_states(data, [0], [1])
    with _closed(state, pools[0][0], data["network"]["colors"]):
        before = _snapshot(state)
        result = learn.measure(_model(data), state, _candidate(), work_budget=1)
        assert result["rejected"] == 1 and result["reward"] < 0
        assert result["work"] == result["gain"] == result["useful"] == 0
        assert _snapshot(state) == before


@pytest.mark.parametrize("saves_nodes", [True, False])
def test_search_reward_uses_actual_dfs_savings_and_excludes_grading_time(monkeypatch, saves_nodes):
    if saves_nodes:
        _, state, _, _ = learn.training_states(_data(), [0], [1])
        for color in state.domains:
            state.restrict_domain(color, (0, 1))
    else:
        relation = np.array([[1, 0, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
        network = ConstraintNetwork({
            "ab": NumpyCore(values=relation.copy(), colors=["a", "b"]),
            "ac": NumpyCore(values=relation.copy(), colors=["a", "c"]),
            "bc": NumpyCore(values=~np.eye(3, dtype=bool), colors=["b", "c"])})
        state = network.create_mac_state(dict(b=(0, 1), c=(0, 1)))
    assert state.propagate()
    model = _model(dict(reward_colors=list(state.domains)))
    before, statistics, rows = _snapshot(state), state.statistics, []

    def search(domains):
        result = state.constraint_network.solve_mac(domains, max_nodes=100)
        assert result.solved and not result.node_limit_reached
        rows.append(result)
        return dict(status="solved", nodes=result.nodes)

    clock = iter((10., 12., 20., 27.))
    monkeypatch.setattr(learn, "perf_counter", lambda: next(clock))
    metrics = learn.measure(model, state, _candidate(), cost_weight=0., proposal_s=.25, search=search)
    assert [row.nodes for row in rows] == ([3, 2] if saves_nodes else [3, 3])
    assert metrics["useful"] and metrics["information"] > 0
    assert metrics["search_gain"] == int(saves_nodes)
    assert metrics["search_log_gain"] == pytest.approx(log(4 / 3) if saves_nodes else 0.)
    assert metrics["cost_s"] == 2.25 and metrics["search_s"] == 7.
    remaining = sum(log(len(values)) for values in before[0].values())
    expected = metrics["information"] / remaining + metrics["search_log_gain"] - .001 - .01 * 2.25
    assert metrics["reward"] == pytest.approx(expected) and not metrics["search_limited"]
    assert _snapshot(state) == before and state.statistics == statistics


@pytest.mark.parametrize("kind", ["skip", "noop", "rejected"])
def test_search_grader_is_not_called_without_actual_pruning(kind):
    data = _data()
    _, state, pools, _ = learn.training_states(data, [0], [1])
    with _closed(state, pools[0][0], data["network"]["colors"]):
        before = _snapshot(state)
        candidate = None if kind == "skip" else SummaryClusterCandidate("edge", ("ab",), ("a",))
        limits = {}
        if kind == "rejected":
            candidate, limits = _candidate(), dict(work_budget=1)
        metrics = learn.measure(_model(data), state, candidate, **limits,
            search=lambda domains: pytest.fail("A no-op must not invoke DFS grading"))
        assert not metrics["useful"] and metrics["search_gain"] == metrics["search_log_gain"] == 0
        assert metrics["search_s"] == 0 and _snapshot(state) == before


@pytest.mark.parametrize("rows", [
    [("node_limit", 1), ("solved", 2)], [("solved", 3), ("wall_timeout", None)]])
def test_censored_search_grading_never_invents_node_savings(rows):
    data = _data()
    _, state, pools, _ = learn.training_states(data, [0], [1])
    answers = iter(dict(status=status, nodes=nodes) for status, nodes in rows)
    with _closed(state, pools[0][0], data["network"]["colors"]):
        before = _snapshot(state)
        metrics = learn.measure(_model(data), state, _candidate(), search=lambda domains: next(answers))
        assert metrics["useful"] and metrics["search_limited"] == 1
        assert metrics["search_gain"] == metrics["search_log_gain"] == 0
        expected = .5 - .001 - .01 * metrics["cost_s"] - .02 * metrics["work"] / 10_000
        assert metrics["reward"] == pytest.approx(expected) and _snapshot(state) == before


def test_remote_seed_augmentation_preserves_pruning_and_skips_unreachable_or_oversized_paths():
    data = _data()
    _, original, _, _ = learn.training_states(data, [0], [1])
    cores = dict(original._cores)
    cores.update({key: NumpyCore(values=np.ones((4, 4), dtype=bool), colors=list(key))
                  for key in ("ax", "xy", "yz")})
    network = ConstraintNetwork(cores, maintain_indices=True)
    state = network.create_mac_state({color: (0, 1) for color in network.dimension_dict})
    assert state.propagate()
    model = Policy(network_data(network), width=8, rounds=1, contextual=True,
                   min_active=2, cyclic_only=True, compact_domains=True, max_cores=6)
    model.reward_colors = list(state.domains)

    class Seed:
        def __init__(self, key):
            self.key = model.key_idx[key]

        def choice(self, items):
            return self.key if self.key in items else items[0]

    before = _snapshot(state)
    candidates = list(learn.mutations(model, state, [_candidate()], Seed("yz"), 1))
    assert len(candidates) == 1
    augmented = candidates[0]
    assert set(augmented.core_keys) == {"ab", "bc", "ac", "ax", "xy", "yz"}
    assert augmented.open_colors == _candidate().open_colors
    baseline = evaluate_candidate(state, _candidate(), compact_domains=True)
    result = evaluate_candidate(state, augmented, compact_domains=True)
    assert baseline.removed_domain_values == result.removed_domain_values == 3
    assert all(set(result.domains[color]) <= set(values) for color, values in baseline.domains.items())
    assert _snapshot(state) == before
    state.push_level()
    assert runtime.apply_candidate(state, augmented, compact_domains=True)
    assert state.snapshot_domains() == result.domains and state.decision_level == 1
    assert not state._temporary_core_keys and not state._added_core_keys
    state.pop_level()
    assert _snapshot(state) == before
    assert not list(learn.mutations(model, state, [_candidate()], Seed("uv"), 1))
    model.config["max_cores"] = 5
    assert not list(learn.mutations(model, state, [_candidate()], Seed("yz"), 1))
    assert _snapshot(state) == before


def test_training_split_excludes_validation_samples_and_rejects_normalized_root_overlap():
    data = _data()
    data["samples"].append(dict(puzzle=0, domains=[[0] for _ in data["network"]["colors"]]))
    _, state, pools, validation = learn.training_states(data, [0], [1])
    assert set(pools) == {0} and len(pools[0]) == 1
    assert validation == [data["initial_domains"][1]]
    assert pools[0][0]["root"]
    assert all(domain == (0, 1) for domain in pools[0][0]["domains"])
    assert state.decision_level == 0 and not state._temporary_core_keys
    duplicate = deepcopy(data)
    duplicate["initial_domains"][1] = {color: [1, 0, 1] for color in duplicate["network"]["colors"]}
    with pytest.raises(ValueError, match="overlap"):
        learn.training_states(duplicate, [0], [1])
    with pytest.raises(ValueError, match="disjoint"):
        learn.training_states(data, [0], [0])


def test_training_gac_preparation_progress_is_optional_and_preserves_data_and_scratch_state(capsys):
    data = _data()
    data["initial_domains"].append({color: list(range(4)) for color in data["network"]["colors"]})
    original = deepcopy(data)
    _, quiet_state, quiet_pools, quiet_validation = learn.training_states(data, [0, 1, 2], [])
    assert capsys.readouterr().out == ""
    _, state, pools, validation = learn.training_states(data, [0, 1, 2], [], progress_every=2)
    log = capsys.readouterr().out
    assert "GAC preparation: 3 training roots; 0 validation instances" in log
    assert "GAC preparation 2/3 roots; unresolved pools 2" in log
    assert "GAC preparation 3/3 roots; unresolved pools 3" in log
    assert pools == quiet_pools and validation == quiet_validation
    assert _snapshot(state) == _snapshot(quiet_state)
    assert state.decision_level == 0 and not state._temporary_core_keys
    assert data == original


@pytest.mark.parametrize("progress_every", [-1, True, 1.5])
def test_training_gac_preparation_rejects_invalid_progress_intervals(progress_every):
    with pytest.raises(ValueError, match="progress interval"):
        learn.training_states(_data(), progress_every=progress_every)


def test_dataset_partitions_are_used_unless_explicit_arguments_override_them():
    data = {**_data(), "train_indices": [1], "validation_indices": [0]}
    _, _, pools, validation = learn.training_states(data)
    assert set(pools) == {1} and validation == [data["initial_domains"][0]]
    assert all(values == (2, 3) for sample in pools[1] for values in sample["domains"])
    _, _, pools, validation = learn.training_states(data, [0], [1])
    assert set(pools) == {0} and validation == [data["initial_domains"][1]]
    _, _, pools, validation = learn.training_states(data, validation_indices=[])
    assert set(pools) == {1} and not validation
    # Each omitted argument retains its stored partition, including its checks.
    with pytest.raises(ValueError, match="disjoint"):
        learn.training_states(data, train_indices=[0])
    collision = {**data, "train_indices": [0]}
    with pytest.raises(ValueError, match="disjoint"):
        learn.training_states(collision)
    assert set(learn.training_states(collision, [0], [1])[2]) == {0}
    with pytest.raises(ValueError, match="valid"):
        learn.training_states({**data, "validation_indices": [2]})
    duplicate = deepcopy(data)
    duplicate["initial_domains"][0] = {color: [3, 2, 3] for color in data["network"]["colors"]}
    with pytest.raises(ValueError, match="overlap"):
        learn.training_states(duplicate)


def test_dataset_without_partition_keeps_the_default_eighty_twenty_split():
    data = _data()
    data["initial_domains"].extend([
        {color: list(((0, 1), (2, 3))[group if color in "abc" else 1 - group])
         for color in data["network"]["colors"]} for group in (0, 1)])
    data["initial_domains"].append({color: list(range(4)) for color in data["network"]["colors"]})
    _, _, pools, validation = learn.training_states(data)
    assert set(pools) == {0, 1, 2, 3}
    assert validation == [data["initial_domains"][4]]


def test_curriculum_shortens_legal_core_prefixes_then_uses_fully_unforced_construction():
    data = _data()
    model = Policy(data["network"], width=8, rounds=1, contextual=True, max_cores=3,
                   max_open_colors=3, min_active=2, cyclic_only=True, heuristic_candidates=0)
    domains = data["initial_domains"][0]
    teachers = [(_candidate(("a", "b", "c")), dict(useful=1))]
    prefixes = [learn.curriculum_prefix(model, domains, teachers, epoch, 8, Random(17))
                for epoch in (0, 3, 5)]
    assert [len(prefix) for prefix in prefixes] == [4, 2, 1]
    assert prefixes[0][-1] == ("finish", -1)
    for prefix in prefixes:
        assert not any(kind == "open" for kind, _ in prefix)
        candidate, log_prob, _, trace = model.rollout(domains, prefix=prefix, beta=0, greedy=True)
        assert trace[:len(prefix)] == prefix and torch.isfinite(log_prob)
        assert {model.keys[idx] for kind, idx in prefix if kind == "core"} <= set(candidate.core_keys)
    for epochs in (4, 8, 12):
        for epoch in range(ceil(.75 * epochs), epochs):
            assert learn.curriculum_prefix(model, domains, teachers, epoch, epochs, Random(1)) is None
    assert learn.curriculum_prefix(model, domains, [], 0, 8, Random(1)) is None


def test_teacher_loss_credits_all_legal_positive_continuations_and_is_order_invariant(monkeypatch):
    data = _data()
    model = Policy(data["network"], width=8, rounds=1, contextual=True, max_cores=3,
                   max_open_colors=3, min_active=2, cyclic_only=True, heuristic_candidates=0)
    teachers = [(_candidate(), dict(useful=1)),
                (SummaryClusterCandidate("other", ("uv", "vw", "uw"), ("u",)), dict(useful=1))]
    logits = torch.zeros(7, requires_grad=True)  # Skip and six possible starting cores.

    def uniform_actions(encoded, sizes, keys, opened, output_phase, beta):
        actions = model.actions(keys, opened, output_phase, sizes)
        scores = logits if not keys else torch.zeros(len(actions))
        return actions, torch.distributions.Categorical(logits=scores)

    monkeypatch.setattr(model, "distribution", uniform_actions)

    class First:
        def choice(self, items):
            return items[0]

        def shuffle(self, items):
            pass

    loss = learn.teacher_loss(model, data["initial_domains"][0], teachers, First())
    # Both triangles' six starting cores are credited (mass 6/7). At output
    # selection only the measured output and then finish count (1/4 and 1/3).
    assert float(loss) == pytest.approx((log(7 / 6) + log(12) / 5) / 2)
    single = learn.teacher_loss(model, data["initial_domains"][0], teachers[:1], First())
    assert float(single - loss) == pytest.approx(log(2) / 2)
    loss.backward()
    assert logits.grad[0] > 0  # Skip is not a useful teacher continuation.
    assert (logits.grad[1:] < 0).all()  # Includes the other, unchosen positive triangle.
    for seed in range(4):
        reversed_order = learn.teacher_loss(model, data["initial_domains"][0], teachers[::-1], Random(seed))
        assert float(reversed_order) == pytest.approx(float(loss))


def test_target_first_teacher_loss_credits_goals_and_only_compatible_core_continuations(monkeypatch):
    closing = np.ones((4, 4), dtype=bool)
    closing[1, 1] = False
    cores = {key: NumpyCore(values=values.copy(), colors=list(key)) for key, values in (
        ("ab", np.eye(4, dtype=bool)), ("bc", np.eye(4, dtype=bool)), ("ac", closing),
        ("bd", np.eye(4, dtype=bool)), ("ad", closing))}
    network = ConstraintNetwork(cores, maintain_indices=True)
    state = network.create_mac_state({color: (0, 1) for color in "abcd"})
    assert state.propagate()
    teachers = [(_candidate(), dict(useful=1)),
                (SummaryClusterCandidate("other", ("ab", "bd", "ad"), ("d",)), dict(useful=1))]
    assert all(evaluate_candidate(state, candidate, compact_domains=True).removed_domain_values == 4
               for candidate, _ in teachers)
    model = Policy(network_data(network), width=8, rounds=1, contextual=True, target_first=True,
                   max_cores=3, max_open_colors=3, min_active=2, cyclic_only=True, heuristic_candidates=0)
    model.reward_colors = list(state.domains)
    goals, seed = torch.zeros(5, requires_grad=True), torch.zeros(4, requires_grad=True)
    observed = []

    def uniform_actions(encoded, sizes, keys, opened, output_phase, beta):
        actions = model.actions(keys, opened, output_phase, sizes)
        observed.append(actions)
        scores = goals if not keys and not opened else seed if not keys else torch.zeros(len(actions))
        return actions, torch.distributions.Categorical(logits=scores)

    class First:
        def choice(self, items):
            return items[0]

        def shuffle(self, items):
            pass

    monkeypatch.setattr(model, "distribution", uniform_actions)
    loss = learn.teacher_loss(model, state.domains, teachers, First())
    # Goal mass is 2/5 (a or d). After choosing a, ad is a legal seed but belongs
    # only to the d teacher, so it must receive no credit for this continuation.
    assert float(loss) == pytest.approx((log(5 / 2) + log(36) / 5) / 2)
    assert observed[1] == [("skip", -1), ("core", model.key_idx["ab"]),
                           ("core", model.key_idx["ac"]), ("core", model.key_idx["ad"])]
    loss.backward()
    assert (goals.grad[[1, 4]] < 0).all() and (goals.grad[[0, 2, 3]] > 0).all()
    assert (seed.grad[[1, 2]] < 0).all() and (seed.grad[[0, 3]] > 0).all()


def test_target_labels_use_actual_pruned_outputs_while_retaining_the_full_summary(monkeypatch):
    data = _data()
    _, original, _, _ = learn.training_states(data, [0], [1])
    network = ConstraintNetwork({**original._cores,
        "ax": NumpyCore(values=np.ones((4, 4), dtype=bool), colors=["a", "x"])}, maintain_indices=True)
    state = network.create_mac_state({color: (0, 1) for color in network.dimension_dict})
    assert state.propagate()
    candidate = SummaryClusterCandidate("extra_output", ("ab", "bc", "ac", "ax"), ("x", "a"))
    model = Policy(network_data(network), width=8, rounds=1, contextual=True, target_first=True,
                   compact_domains=True, min_active=2, cyclic_only=True, max_cores=4, heuristic_candidates=0)
    model.reward_colors = list(state.domains)
    before = _snapshot(state)
    metrics = learn.measure(model, state, candidate, known_satisfiable=True)
    assert metrics["useful"] and metrics["gain"] == 3 and metrics["pruned_outputs"] == ("a",)
    assert _snapshot(state) == before
    teachers = [(candidate, metrics)]
    for seed in range(8):
        trace = learn._teacher_trace(model, state.domains, teachers[0], Random(seed))
        assert trace[0] == ("open", model.color_idx["a"])
        constructed, _, _, _ = model.rollout(state.domains, beta=0., trace=trace)
        assert set(constructed.open_colors) == {"a", "x"}
        prefix = learn.curriculum_prefix(model, state.domains, teachers, 0, 8, Random(seed))
        assert prefix[0] == trace[0]
    goals = torch.zeros(len(state.domains) + 1, requires_grad=True)

    def uniform_actions(encoded, sizes, keys, opened, output_phase, beta):
        actions = model.actions(keys, opened, output_phase, sizes)
        scores = goals if not keys and not opened else torch.zeros(len(actions))
        return actions, torch.distributions.Categorical(logits=scores)

    monkeypatch.setattr(model, "distribution", uniform_actions)
    learn.teacher_loss(model, state.domains, teachers, Random(3)).backward()
    assert goals.grad[1 + model.color_idx["a"]] < 0
    assert goals.grad[1 + model.color_idx["x"]] > 0  # Unchanged x is not an initial target.
    state.push_level()
    assert runtime.apply_candidate(state, candidate, compact_domains=True)
    assert state.domains["a"] == (0,) and state.domains["x"] == (0, 1)
    assert not state._temporary_core_keys and not state._added_core_keys
    state.pop_level()
    assert _snapshot(state) == before


def _on_policy_problem(**limits):
    closing = np.ones((4, 4), dtype=bool)
    closing[1, 1] = False
    network = ConstraintNetwork({key: NumpyCore(values=values, colors=list(key)) for key, values in (
        ("ab", np.eye(4, dtype=bool)), ("bc", np.eye(4, dtype=bool)), ("ac", closing),
        ("ax", np.ones((4, 4), dtype=bool)))}, maintain_indices=True)
    state = network.create_mac_state({color: (0, 1) for color in network.dimension_dict})
    assert state.propagate()
    config = dict(width=8, rounds=1, contextual=True, target_first=True, compact_domains=True,
                  min_active=2, cyclic_only=False, max_cores=4, max_open_colors=2,
                  max_work=1000, max_size=16, heuristic_candidates=0)
    model = Policy(network_data(network), **{**config, **limits})
    model.reward_colors = list(state.domains)
    candidate = _candidate()
    teachers = [(candidate, learn.measure(model, state, candidate, known_satisfiable=True))]
    assert teachers[0][1]["useful"] and teachers[0][1]["pruned_outputs"] == ("a",)
    return model, state, teachers


def test_on_policy_imitation_repairs_student_prefix_without_crediting_useless_finishes(monkeypatch):
    model, state, teachers = _on_policy_problem()
    before, observed = _snapshot(state), []

    def student(encoded, sizes, keys, opened, output_phase, beta):
        actions = model.actions(keys, opened, output_phase, sizes)
        preferred = {("finish", -1): 15., ("open", model.color_idx["a"]): 20.,
                     ("core", model.key_idx["ax"]): 20., ("core", model.key_idx["ab"]): 14.,
                     ("core", model.key_idx["bc"]): 13., ("core", model.key_idx["ac"]): 12.}
        if output_phase:
            preferred[("open", model.color_idx["x"])] = 20.
        scores = torch.tensor([preferred.get(action, -6.) for action in actions], requires_grad=True)
        observed.append((tuple(keys), tuple(opened), output_phase, actions, scores))
        return actions, torch.distributions.Categorical(logits=scores)

    monkeypatch.setattr(model, "distribution", student)
    with monkeypatch.context() as no_trials:
        no_trials.setattr(learn, "evaluate_candidate", lambda *a, **k: pytest.fail("Repair must not contract"))
        no_trials.setattr(runtime, "contract_candidate", lambda *a, **k: pytest.fail("Repair must only plan"))
        loss = learn.on_policy_teacher_loss(model, state, teachers)
        assert torch.isfinite(loss)
        loss.backward()
    ax = model.key_idx["ax"]
    # The seed ax is outside the measured teacher. Its own prefix is trained,
    # while its preferred early finish is corrected into the useful triangle.
    prefix = next(item for item in observed if item[0] == (ax,) and not item[2])
    assert prefix[4].grad[prefix[3].index(("finish", -1))] > 0
    assert set(observed[-1][0]) == {model.key_idx[key] for key in ("ab", "bc", "ac", "ax")}
    assert set(observed[-1][1]) == {model.color_idx[color] for color in ("a", "x")}
    assert observed[-1][2]  # Finished output selection after retaining a.
    repaired = SummaryClusterCandidate("repaired", tuple(model.keys[key] for key in observed[-1][0]),
        tuple(model.network["colors"][color] for color in observed[-1][1]))
    result = evaluate_candidate(state, repaired, compact_domains=True, max_work=1000, max_size=16)
    assert result.removed_domain_values == 3 and result.domains["a"] == (0,)
    assert result.domains["x"] == (0, 1)
    assert _snapshot(state) == before and not state._temporary_core_keys


@pytest.mark.parametrize("limits,work_budget", [({"max_cores": 3}, None), ({}, 12)])
def test_on_policy_imitation_rejects_student_extensions_without_affordable_completion(monkeypatch, limits, work_budget):
    model, state, teachers = _on_policy_problem(**limits)
    before, observed = _snapshot(state), []

    def student(encoded, sizes, keys, opened, output_phase, beta):
        actions = model.actions(keys, opened, output_phase, sizes)
        scores = torch.zeros(len(actions))
        preferred = ("open", model.color_idx["a"]) if not keys and not opened else ("core", model.key_idx["ax"])
        if preferred in actions:
            scores[actions.index(preferred)] = 10.
        scores.requires_grad_()
        observed.append((tuple(keys), tuple(opened), output_phase, actions, scores))
        return actions, torch.distributions.Categorical(logits=scores)

    monkeypatch.setattr(model, "distribution", student)
    loss = learn.on_policy_teacher_loss(model, state, teachers, work_budget=work_budget)
    assert torch.isfinite(loss)
    loss.backward()
    seed = next(item for item in observed if not item[0] and item[1])
    assert seed[4].grad[seed[3].index(("core", model.key_idx["ax"]))] > 0
    assert all(model.key_idx["ax"] not in item[0] for item in observed)
    assert _snapshot(state) == before


def test_on_policy_imitation_retains_pruned_goal_and_expanded_output_budget(monkeypatch):
    model, state, teachers = _on_policy_problem(max_size=4)
    before, observed = _snapshot(state), []

    def student(encoded, sizes, keys, opened, output_phase, beta):
        actions = model.actions(keys, opened, output_phase, sizes)
        # x is eligible but is not a measured pruning target. After correction
        # to a, its expanded second output axis cannot fit the four-entry cap.
        scores = torch.tensor([10. if action == ("open", model.color_idx["x"]) else 0.
                               for action in actions], requires_grad=True)
        observed.append((tuple(keys), tuple(opened), output_phase, actions, scores))
        return actions, torch.distributions.Categorical(logits=scores)

    monkeypatch.setattr(model, "distribution", student)
    loss = learn.on_policy_teacher_loss(model, state, teachers)
    assert torch.isfinite(loss)
    loss.backward()
    assert observed[0][4].grad[observed[0][3].index(("open", model.color_idx["x"]))] > 0
    assert all(not opened or opened[0] == model.color_idx["a"] for _, opened, _, _, _ in observed)
    assert any(output_phase for _, _, output_phase, _, _ in observed)
    assert all(("open", model.color_idx["x"]) not in actions
               for _, _, output_phase, actions, _ in observed if output_phase)
    assert _snapshot(state) == before


def test_on_policy_imitation_requires_an_explicit_target_first_policy(tmp_path):
    model, state, teachers = _on_policy_problem(target_first=False)
    before = _snapshot(state)
    with pytest.raises(ValueError, match="target-first"):
        learn.on_policy_teacher_loss(model, state, teachers)
    with pytest.raises(ValueError, match="target-first"):
        learn.fit(_data(), output=tmp_path / "invalid.pt", on_policy_imitation=True)
    assert _snapshot(state) == before and not list(tmp_path.iterdir())


def test_completion_value_labels_best_terminal_utility_including_near_ties(monkeypatch):
    data = _data()
    _, state, pools, _ = learn.training_states(data, [0], [1])
    model = Policy(data["network"], width=8, rounds=1, contextual=True, target_first=True,
                   max_cores=3, max_open_colors=1, heuristic_candidates=0)
    teachers = [(_candidate(), dict(reward=.2, pruned_outputs=("a",))),
        (SummaryClusterCandidate("better", ("uv", "vw", "uw"), ("u",)),
         dict(reward=.8, pruned_outputs=("u",))),
        (SummaryClusterCandidate("near_tie", ("uv", "vw", "uw"), ("v",)),
         dict(reward=.79995, pruned_outputs=("v",)))]
    logits = torch.tensor([1.] + [0.] * len(model.color_idx), requires_grad=True)
    calls = []

    def stop(encoded, sizes, keys, opened, phase, beta):
        actions = model.actions(keys, opened, phase, sizes)
        calls.append((keys, opened, phase))
        return actions, torch.distributions.Categorical(logits=logits)

    monkeypatch.setattr(model, "distribution", stop)
    with _closed(state, pools[0][0], data["network"]["colors"]):
        before = _snapshot(state)
        assert all(evaluate_candidate(state, candidate).removed_domain_values == 3 for candidate, _ in teachers)
        loss = learn.completion_value_loss(model, state.domains, teachers)
        assert float(loss) == pytest.approx(float(logits.logsumexp(0)) - log(2))
        loss.backward()
        assert logits.grad[1 + model.color_idx["u"]] < 0 and logits.grad[1 + model.color_idx["v"]] < 0
        assert logits.grad[1 + model.color_idx["a"]] > 0  # Positive, but below the best terminal utility.
        assert logits.grad[0] > 0 and len(calls) == 1  # Student skip is followed without correction.
        assert torch.isfinite(logits.grad).all() and _snapshot(state) == before


@pytest.mark.parametrize("seed", ["ax", "ab"])
def test_completion_value_follows_dead_or_prematurely_finished_student_prefix_without_inventing_labels(monkeypatch, seed):
    model, state, teachers = _on_policy_problem()
    before, observed = _snapshot(state), []

    def student(encoded, sizes, keys, opened, phase, beta):
        actions = model.actions(keys, opened, phase, sizes)
        desired = (("open", model.color_idx["a"]) if not opened else
                   ("core", model.key_idx[seed]) if not keys else ("finish", -1))
        logits = torch.tensor([10. if action == desired else 0. for action in actions], requires_grad=True)
        observed.append((tuple(keys), tuple(opened), phase, actions, logits))
        return actions, torch.distributions.Categorical(logits=logits)

    monkeypatch.setattr(model, "distribution", student)
    monkeypatch.setattr(learn, "evaluate_candidate", lambda *a, **k: pytest.fail("Labels must not run new trials"))
    loss = learn.completion_value_loss(model, state.domains, teachers)
    assert torch.isfinite(loss)
    loss.backward()
    assert len(observed) == 4 and observed[2][0] == observed[3][0] == (model.key_idx[seed],)
    assert not observed[2][2] and observed[3][2]  # Its premature finish is actually followed.
    assert observed[3][4].grad is None  # No measured completion exists in this output prefix.
    if seed == "ax":
        assert observed[2][4].grad is None  # Off-teacher core: no invented repaired completion.
        assert observed[1][4].grad[observed[1][3].index(("core", model.key_idx["ax"]))] > 0
    else:
        assert observed[2][4].grad[observed[2][3].index(("finish", -1))] > 0
    assert all(torch.isfinite(item[4].grad).all() for item in observed if item[4].grad is not None)
    assert _snapshot(state) == before


def test_completion_value_uses_best_still_reachable_utility_after_student_takes_worse_branch(monkeypatch):
    closing = np.array([[1, 1], [1, 0]], dtype=bool)
    network = ConstraintNetwork({key: NumpyCore(values=values, colors=list(key)) for key, values in (
        ("ab", np.eye(2, dtype=bool)), ("bc", np.eye(2, dtype=bool)), ("ac", closing),
        ("bd", np.eye(2, dtype=bool)), ("ad", closing))}, maintain_indices=True)
    state = network.create_mac_state()
    assert state.propagate()
    model = Policy(network_data(network), width=8, rounds=1, contextual=True, target_first=True,
                   max_cores=3, max_open_colors=1, heuristic_candidates=0)
    teachers = [(_candidate(), dict(reward=.2, pruned_outputs=("a",))),
        (SummaryClusterCandidate("better", ("ab", "bd", "ad"), ("a",)),
         dict(reward=.8, pruned_outputs=("a",)))]
    assert all(evaluate_candidate(state, item).removed_domain_values == 4 for item, _ in teachers)
    before, observed = _snapshot(state), []

    def student(encoded, sizes, keys, opened, phase, beta):
        actions = model.actions(keys, opened, phase, sizes)
        desired = (("open", model.color_idx["a"]) if not opened else
                   ("core", model.key_idx["ac"]) if not keys else
                   next((("core", model.key_idx[key]) for key in ("ab", "bc")
                         if model.key_idx[key] not in keys), ("finish", -1)))
        logits = torch.tensor([10. if action == desired else 0. for action in actions], requires_grad=True)
        observed.append((tuple(keys), actions, logits))
        return actions, torch.distributions.Categorical(logits=logits)

    monkeypatch.setattr(model, "distribution", student)
    learn.completion_value_loss(model, state.domains, teachers).backward()
    seed = observed[1]
    assert seed[2].grad[seed[1].index(("core", model.key_idx["ac"]))] > 0
    worse = next(item for item in observed if item[0] == (model.key_idx["ac"],))
    assert all(worse[2].grad[worse[1].index(("core", model.key_idx[key]))] < 0 for key in ("ab", "bc"))
    assert _snapshot(state) == before


def test_completion_value_imitation_guards_and_one_epoch_fit(tmp_path):
    model, state, teachers = _on_policy_problem(target_first=False)
    with pytest.raises(ValueError, match="target-first"):
        learn.completion_value_loss(model, state.domains, teachers)
    model.config["target_first"] = True
    with pytest.raises(ValueError, match="positive measured teacher"):
        learn.completion_value_loss(model, state.domains, [(teachers[0][0], {**teachers[0][1], "reward": 0.})])
    for options in ({}, dict(target_first=True, on_policy_imitation=True)):
        with pytest.raises(ValueError, match="excludes repair imitation"):
            learn.fit(_data(), output=tmp_path / "invalid.pt", completion_value_imitation=True, **options)
    assert not list(tmp_path.iterdir())
    output = tmp_path / "completion.pt"
    model, history = learn.fit(_data(), output=output, epochs=1, width=8, rounds=1, seed=7,
        rollouts=2, steps=1, warmup=1, teacher_candidates=8, teacher_mutations=0,
        train_indices=[0], validation_indices=[1], validation_seconds=2., max_cores=3,
        max_open_colors=3, max_work=10_000, max_size=256, target_first=True,
        value_encoding=True, completion_value_imitation=True)
    assert len(history) == 1 and history[0]["updates"] == 1 and history[0]["useful"] > 0
    assert np.isfinite(history[0]["imitation_loss"])
    checkpoint = torch.load(output, weights_only=True)
    assert checkpoint["training"]["completion_value_imitation"] and not checkpoint["training"]["on_policy_imitation"]
    assert runtime.load(output).config["value_encoding"]
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_total_work_budget_is_rejected_before_writing(tmp_path, budget):
    with pytest.raises(ValueError, match="Invalid learning parameters"):
        learn.fit(_data(), output=tmp_path / "invalid.pt", max_total_work=budget)
    assert not list(tmp_path.iterdir())


def test_initialization_preserves_contextual_weights_and_rejects_inherited_validation_overlap(tmp_path):
    data = _data()
    parent_model = Policy(data["network"], width=8, rounds=1, contextual=True,
                          max_cores=3, max_open_colors=3, min_active=2, cyclic_only=True,
                          compact_domains=True, max_work=10_000, max_size=256)
    parent = dict(version=1, kind="construction", network=data["network"], config=parent_model.config,
                  reward_colors=data["reward_colors"], beta=7., weights=parent_model.state_dict(),
                  training=dict(train_puzzles=[0], train_instance_hashes=sorted(
                      learn._instance_hashes(data, [data["initial_domains"][0]]))))
    source, output = tmp_path / "parent.pt", tmp_path / "child.pt"
    torch.save(parent, source)
    source_bytes = source.read_bytes()
    options = dict(epochs=1, width=8, rounds=1, rollouts=2, steps=1, warmup=1,
                   teacher_candidates=8, teacher_mutations=0, train_indices=[0], validation_indices=[1],
                   validation_seconds=2., max_cores=3, max_open_colors=3, max_work=10_000, max_size=256)
    learn.fit(data, output=output, initial_checkpoint=source, **options)
    initial = torch.load(output.with_stem("child.initial"), weights_only=True)
    assert initial["beta"] == 0.
    assert initial["training"]["parent"] == dict(path=str(source), sha256=sha256(source_bytes).hexdigest())
    for key, value in parent["weights"].items():
        torch.testing.assert_close(initial["weights"][key], value, atol=0, rtol=0)
    assert source.read_bytes() == source_bytes
    untracked = deepcopy(parent)
    untracked["training"].pop("train_instance_hashes")
    legacy = tmp_path / "untracked.pt"
    torch.save(untracked, legacy)
    assert runtime.load(legacy).beta == 7.  # Existing inference remains supported.
    with pytest.raises(ValueError, match="lacks training instance fingerprints"):
        learn.fit(data, output=tmp_path / "untracked_child.pt", initial_checkpoint=legacy, **options)
    parent["training"]["train_puzzles"] = [1]
    parent["training"]["train_instance_hashes"] = sorted(
        learn._instance_hashes(data, [data["initial_domains"][1]]))
    leaked = tmp_path / "leaked.pt"
    torch.save(parent, leaked)
    with pytest.raises(ValueError, match="trained on validation puzzles"):
        learn.fit(data, output=tmp_path / "blocked.pt", initial_checkpoint=leaked, **options)
    assert not list(tmp_path.glob("blocked*.pt"))
    inherited = tmp_path / "inherited.pt"
    learn.fit(data, output=inherited, initial_checkpoint=source,
              **{**options, "train_indices": [1], "validation_indices": []})
    with pytest.raises(ValueError, match="trained on validation puzzles"):
        learn.fit(data, output=tmp_path / "grandchild.pt", initial_checkpoint=inherited,
                  **{**options, "train_indices": [1], "validation_indices": [0]})
    assert not list(tmp_path.glob("grandchild*.pt"))


def test_cross_network_fit_requires_explicit_value_encoded_transfer_and_preserves_source_weights(tmp_path):
    data = _data()
    parent = Policy(data["network"], width=8, rounds=1, contextual=True, value_encoding=True,
                    target_first=True, compact_domains=True, max_cores=3, max_open_colors=1)
    source = tmp_path / "parent.pt"
    torch.save(dict(version=1, kind="construction", network=parent.network, config=parent.config,
        reward_colors=data["reward_colors"], beta=7., weights=parent.state_dict(),
        training=dict(train_puzzles=[0], train_instance_hashes=sorted(
            learn._instance_hashes(data, [data["initial_domains"][0]])))), source)
    source_bytes = source.read_bytes()
    larger = deepcopy(data)
    larger["network"]["dimensions"] = [9] * len(parent.network["colors"])
    for core in larger["network"]["cores"]:
        values = np.ones((9, 9), dtype=bool)
        values[:4, :4] = np.asarray(core["values"]).reshape(core["shape"])
        core["shape"], core["values"] = [9, 9], values.ravel().astype(int).tolist()
    options = dict(epochs=1, width=8, rounds=1, rollouts=2, steps=1, warmup=1,
                   teacher_candidates=8, teacher_mutations=0, train_indices=[0], validation_indices=[1],
                   validation_seconds=2., max_cores=3, max_open_colors=1, max_work=10_000,
                   max_size=256, target_first=True, value_encoding=True)
    with pytest.raises(ValueError, match="same tensor network"):
        learn.fit(larger, output=tmp_path / "strict.pt", initial_checkpoint=source, **options)
    with pytest.raises(ValueError, match="value encoding and an initialization checkpoint"):
        learn.fit(larger, output=tmp_path / "missing.pt", transfer_weights=True, **options)
    with pytest.raises(ValueError, match="value encoding and an initialization checkpoint"):
        learn.fit(larger, output=tmp_path / "disabled.pt", initial_checkpoint=source,
                  transfer_weights=True, **{**options, "value_encoding": False})
    legacy = tmp_path / "legacy.pt"
    legacy_model = Policy(data["network"], width=8, rounds=1, contextual=True)
    torch.save(dict(network=legacy_model.network, config=legacy_model.config), legacy)
    with pytest.raises(ValueError, match="value-encoded initialization checkpoint"):
        learn.fit(larger, output=tmp_path / "legacy_child.pt", initial_checkpoint=legacy,
                  transfer_weights=True, **options)
    output = tmp_path / "transferred.pt"
    model, history = learn.fit(larger, output=output, initial_checkpoint=source, transfer_weights=True, **options)
    assert len(history) == 1 and history[0]["useful"] > 0
    initial = torch.load(output.with_stem("transferred.initial"), weights_only=True)
    assert initial["network"] == larger["network"] and initial["beta"] == 0.
    assert initial["training"]["parent"] == dict(path=str(source), sha256=sha256(source_bytes).hexdigest(),
                                                cross_network=True)
    for key, value in parent.state_dict().items():
        torch.testing.assert_close(initial["weights"][key], value, atol=0, rtol=0)
    assert model.config["value_encoding"] and source.read_bytes() == source_bytes
    assert not any((tmp_path / name).exists() for name in
                   ("strict.pt", "missing.pt", "disabled.pt", "legacy_child.pt"))


def test_warmstart_tracks_problem_identity_after_dataset_reordering_and_across_generations(tmp_path):
    data = _data()
    options = dict(epochs=1, width=8, rounds=1, rollouts=2, steps=1, warmup=1,
                   teacher_candidates=8, teacher_mutations=0, validation_seconds=2.,
                   max_cores=3, max_open_colors=3, max_work=10_000, max_size=256)
    source = tmp_path / "original.pt"
    learn.fit(data, output=source, train_indices=[0], validation_indices=[1], **options)
    reordered = deepcopy(data)
    reordered["initial_domains"].reverse()
    reordered["samples"] = [dict(puzzle=idx, domains=[domains[color] for color in data["network"]["colors"]])
                            for idx, domains in enumerate(reordered["initial_domains"])]
    # Previously trained problem 0 is now problem 1: validation on the other
    # problem remains valid, even though its new index matches the old train index.
    learn.fit(reordered, output=tmp_path / "valid.pt", initial_checkpoint=source,
              train_indices=[1], validation_indices=[0], **options)
    with pytest.raises(ValueError, match="trained on validation puzzles"):
        learn.fit(reordered, output=tmp_path / "leak.pt", initial_checkpoint=source,
                  train_indices=[0], validation_indices=[1], **options)
    child = tmp_path / "child.pt"
    learn.fit(reordered, output=child, initial_checkpoint=source,
              train_indices=[0], validation_indices=[], **options)
    with pytest.raises(ValueError, match="trained on validation puzzles"):
        learn.fit(reordered, output=tmp_path / "descendant.pt", initial_checkpoint=child,
                  train_indices=[0], validation_indices=[1], **options)
    assert not list(tmp_path.glob("leak*.pt")) and not list(tmp_path.glob("descendant*.pt"))


def test_checkpoint_selection_does_not_treat_timeout_as_zero_search_nodes(tmp_path, monkeypatch):
    validation = iter((dict(status="node_limit", nodes=100, solve_s=1.),
                       dict(status="wall_timeout", nodes=None, solve_s=1.)))
    monkeypatch.setattr(learn, "evaluate_roots", lambda *a, **k: dict(
        contradictions=0, additional_resolved_fraction=0., mean_progress=.25, mean_seconds=1., rows=[]))

    def search(network, domains, mode, model, **options):
        if mode == "trained":
            return next(validation)
        result = network.solve_mac(domains, max_nodes=100)
        assert result.solved
        return dict(status="solved", nodes=result.nodes, setup_s=0., solve_s=0.)

    monkeypatch.setattr(learn, "solve", search)
    path = tmp_path / "censored.pt"
    learn.fit(_data(), output=path, epochs=1, width=8, rounds=1, rollouts=2, steps=1,
              warmup=1, teacher_candidates=8, teacher_mutations=0, target_first=True,
              train_indices=[0], validation_indices=[1], validation_seconds=2., search_nodes=100,
              max_cores=3, max_open_colors=3, max_work=10_000, max_size=256)
    best = torch.load(path.with_stem("censored.best"), weights_only=True)
    latest = torch.load(path, weights_only=True)
    assert best["history"] == [] and len(latest["history"]) == 1
    assert best["selection"]["search_complete"] == latest["selection"]["search_complete"] == 0
    assert best["selection"]["search_nodes"] is latest["selection"]["search_nodes"] is None


def test_reordered_structural_teacher_is_not_reported_as_novel(tmp_path, monkeypatch):
    candidate = _candidate(("a", "b", "c"))
    original = Policy.rollout

    def sample_measured_teacher(self, domains, **options):
        if self.training and options.get("trace") is None and not options.get("greedy"):
            options.pop("prefix", None)
            options["trace"] = self.candidate_trace(candidate, domains)
        return original(self, domains, **options)

    monkeypatch.setattr(Policy, "rollout", sample_measured_teacher)
    _, history = learn.fit(_data(), output=tmp_path / "reordered.pt", epochs=1, width=8,
        rounds=1, rollouts=2, teacher_candidates=8, teacher_mutations=0, steps=1,
        train_indices=[0], validation_indices=[1], validation_seconds=2., max_cores=3,
        max_open_colors=3, max_work=10_000, max_size=256, curriculum=False)
    row = history[0]
    assert row["useful"] == row["trials"] == 2
    assert row["novel"] == row["useful_novel"] == row["useful_unseen"] == 0


@pytest.mark.parametrize("target_first,search_nodes,on_policy_imitation", [
    (False, 0, False), (True, 0, False), (True, 100, False), (True, 0, True)])
def test_tiny_training_measures_teachers_collects_updated_domains_and_saves_loadable_checkpoints(tmp_path, monkeypatch,
                                                                                             target_first, search_nodes,
                                                                                             on_policy_imitation):
    monkeypatch.setattr(learn, "causal_candidates", lambda *a, **k: pytest.fail("Default fit must not probe"))
    data = _data()
    base, state, pools, validation = learn.training_states(data, [0], [1])
    original = _snapshot(state)
    monkeypatch.setattr(learn, "training_states", lambda *a, **k: (base, state, pools, validation))
    path = tmp_path / "useful.pt"
    model, history = learn.fit(data, output=path, epochs=2, width=8, rounds=1, seed=7,
        rollouts=2, teacher_candidates=8, teacher_mutations=0, steps=1, warmup=2,
        train_indices=[0], validation_indices=[1], validation_seconds=2., max_cores=3,
        max_open_colors=3, max_work=10_000, max_size=256, target_first=target_first,
        search_nodes=search_nodes, search_seconds=2., on_policy_imitation=on_policy_imitation)
    assert len(history) == 2 and all(row["updates"] == 1 for row in history)
    assert all("teacher_probe_assumptions" not in row for row in history)
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    for row in history:
        for key in ("novel", "useful_novel", "useful_unseen", "useful_unforced", "prefixed"):
            assert key in row and 0 <= row[key] <= row["trials"]
        assert row["useful_unseen"] <= row["useful_novel"] <= row["useful"]
        assert row["useful_unforced"] <= row["useful"] and row["novel"] <= row["selected"]
        if not row["useful"]:
            assert row["reward"] < 0
    assert sum(row["teacher_trials"] for row in history) > 0
    if search_nodes:
        assert sum(row["search_gain"] for row in history) > 0
        assert history[-1]["search_grading"]["calls"] > 0
    assert all(row["validation"]["source"] == "validation_roots" for row in history)
    assert len(pools[0]) > 1
    advanced = [sample for sample in pools[0] if not sample["root"]]
    assert advanced and any(len(values) == 1 for values in advanced[0]["domains"])
    assert any(len(values) > 1 for values in advanced[0]["domains"])
    assert _snapshot(state) == original and not state._temporary_core_keys
    for checkpoint_path in (path, path.with_stem("useful.best"), path.with_stem("useful.initial")):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        assert checkpoint["beta"] == 0. and checkpoint["config"]["compact_domains"]
        assert checkpoint["config"]["contextual"]
        assert bool(checkpoint["config"].get("target_first")) == target_first
        assert checkpoint["cost_metric"] == "conditioned_contraction_work"
        assert checkpoint["training"]["method"] == "immediate_construction"
        assert checkpoint["training"]["train_puzzles"] == [0]
        if checkpoint["history"]:
            assert checkpoint["training"]["on_policy_imitation"] == on_policy_imitation
        assert load(checkpoint_path).beta == runtime.load(checkpoint_path).beta == 0.
    initial = load(path.with_stem("useful.initial"))
    assert any(not torch.equal(value, initial.state_dict()[key]) for key, value in model.state_dict().items())
    with _closed(state, pools[0][0], model.network["colors"]), torch.no_grad():
        candidates = [_candidate(("a", "b", "c")),
                      SummaryClusterCandidate("other", ("uv", "vw", "uw"), ("u", "v", "w"))]
        likelihood = lambda policy: sum(float(policy.rollout(state.domains, beta=0.,
            trace=policy.candidate_trace(candidate, state.domains))[1]) for candidate in candidates)
        assert likelihood(model) > likelihood(initial)
    before = {item: item.read_bytes() for item in tmp_path.iterdir()}
    with pytest.raises(FileExistsError, match="already exists"):
        learn.fit(data, output=path, epochs=1)
    assert {item: item.read_bytes() for item in tmp_path.iterdir()} == before


@pytest.mark.parametrize("after_nodes", [2, 5, 9])
def test_node_objective_uses_raw_completed_savings_even_when_pruning_increases_search(after_nodes):
    data = _data()
    _, state, pools, _ = learn.training_states(data, [0], [1])
    answers = iter((dict(status="solved", nodes=5), dict(status="solved", nodes=after_nodes)))
    with _closed(state, pools[0][0], data["network"]["colors"]):
        before = _snapshot(state)
        result = learn.measure(_model(data), state, _candidate(), objective="dfs_nodes",
            cost_weight=100., solve_reward=100., contradiction_reward=100., proposal_s=100.,
            search=lambda domains: next(answers), known_satisfiable=True)
        assert result["useful"] and result["information"] > 0 and result["work"] > 0
        assert result["reward_valid"] == 1 and not result["search_limited"]
        assert result["reward"] == result["search_gain"] == 5 - after_nodes
        assert _snapshot(state) == before


@pytest.mark.parametrize("contradiction", [False, True])
def test_node_objective_has_no_extra_solve_or_contradiction_bonus(contradiction):
    if contradiction:
        cores = {key: NumpyCore(values=~np.eye(3, dtype=bool), colors=list(key))
                 for key in ("ab", "bc", "ac")}
        state = ConstraintNetwork(cores).create_mac_state({color: (0, 1) for color in "abc"})
    else:
        _, state, _, _ = learn.training_states(_data(), [0], [1])
        for color in state.domains:
            state.restrict_domain(color, (0, 1) if color in "abc" else (0,))
    assert state.propagate() and not state.is_solved()
    before = _snapshot(state)
    status = "unsat" if contradiction else "solved"
    answers = iter((dict(status=status, nodes=7), dict(status=status, nodes=1)))
    result = learn.measure(_model(dict(reward_colors=list(state.domains))), state, _candidate(),
        objective="dfs_nodes", solve_reward=100., contradiction_reward=100., cost_weight=100.,
        search=lambda domains: next(answers))
    assert bool(result["contradictions"]) == contradiction
    assert bool(result["solved"]) == (not contradiction)
    assert result["reward_valid"] == 1 and result["reward"] == result["search_gain"] == 6
    assert _snapshot(state) == before


@pytest.mark.parametrize("statuses", [("node_limit", "solved"), ("solved", "wall_timeout"),
                                     ("node_limit", "node_limit")])
def test_node_objective_marks_censored_counterfactuals_unknown(statuses):
    data = _data()
    _, state, pools, _ = learn.training_states(data, [0], [1])
    answers = iter(dict(status=status, nodes=None if status == "wall_timeout" else 10)
                   for status in statuses)
    with _closed(state, pools[0][0], data["network"]["colors"]):
        before = _snapshot(state)
        result = learn.measure(_model(data), state, _candidate(), objective="dfs_nodes",
                               search=lambda domains: next(answers))
        assert result["useful"] and result["search_limited"] == 1
        assert result["reward_valid"] == 0 and result["reward"] == 0
        assert not learn._positive_teachers([(_candidate(), result)], objective="dfs_nodes")
        assert _snapshot(state) == before


@pytest.mark.parametrize("kind", ["skip", "noop", "rejected"])
def test_node_objective_unchanged_domains_have_zero_reward_without_running_dfs(kind):
    data = _data()
    _, state, pools, _ = learn.training_states(data, [0], [1])
    candidate = (None if kind == "skip" else SummaryClusterCandidate("edge", ("ab",), ("a",))
                 if kind == "noop" else _candidate())
    with _closed(state, pools[0][0], data["network"]["colors"]):
        before = _snapshot(state)
        result = learn.measure(_model(data), state, candidate, objective="dfs_nodes",
            search=lambda domains: pytest.fail("An unchanged state needs no counterfactual solve"),
            **(dict(work_budget=1) if kind == "rejected" else {}))
        assert result["reward"] == 0 and result["reward_valid"] == 1 and not result["useful"]
        assert bool(result["rejected"]) == (kind == "rejected")
        assert _snapshot(state) == before


def test_node_teachers_require_verified_positive_savings_without_a_pruning_fallback():
    def item(name, gain, *, valid=1, reward=None, useful=1):
        return (SummaryClusterCandidate(name, ("ab", "bc", "ac"), ("a",)),
                dict(useful=useful, reward=gain if reward is None else reward,
                     search_gain=gain, reward_valid=valid))

    saving = item("saving", 2)
    alternatives = [item("pruning_only", 0, reward=1), item("harmful", -1, reward=1),
                    item("censored", 100, valid=0), item("not_useful", 100, useful=0)]
    assert learn._positive_teachers([saving, *alternatives], objective="dfs_nodes") == [saving]
    assert not learn._positive_teachers(alternatives, search=True, objective="dfs_nodes")
    # The previous mixed objective retains its useful-pruning fallback.
    assert learn._positive_teachers([alternatives[0]], search=True) == [alternatives[0]]


@pytest.mark.parametrize("options", [dict(objective="unknown"), dict(objective="dfs_nodes"),
    dict(objective="dfs_nodes", search_nodes=100, validation_indices=[]),
    dict(objective="dfs_nodes", search_nodes=100, target_first=True, on_policy_imitation=True)])
def test_invalid_node_learning_setup_does_not_write_checkpoints(tmp_path, options):
    with pytest.raises(ValueError):
        learn.fit(_data(), output=tmp_path / "invalid.pt", **options)
    assert not list(tmp_path.iterdir())


def test_node_objective_tiny_fit_saves_deployment_budget_and_preserves_source_state(tmp_path, monkeypatch):
    data = _data()
    base, state, pools, validation = learn.training_states(data, [0], [1])
    before = _snapshot(state)
    monkeypatch.setattr(learn, "training_states", lambda *a, **k: (base, state, pools, validation))
    output = tmp_path / "nodes.pt"
    model, history = learn.fit(data, output=output, objective="dfs_nodes", epochs=1, width=8,
        rounds=1, seed=7, rollouts=2, steps=1, warmup=1, teacher_candidates=8, teacher_mutations=0,
        train_indices=[0], validation_indices=[1], validation_seconds=5., search_nodes=100,
        search_seconds=5., max_total_work=None, cluster_attempts=512, max_cores=3,
        max_open_colors=3, max_work=10_000, max_size=256, target_first=True,
        value_encoding=True, completion_value_imitation=True)
    assert len(history) == 1 and history[0]["updates"] == 1
    assert history[0]["validation"]["search_complete"] == 1
    assert history[0]["validation"]["search_nodes"] is not None
    assert _snapshot(state) == before and not state._temporary_core_keys
    initial = load(output.with_stem("nodes.initial"))
    assert any(not torch.equal(value, initial.state_dict()[key]) for key, value in model.state_dict().items())
    for path in (output, output.with_stem("nodes.best"), output.with_stem("nodes.initial")):
        checkpoint = torch.load(path, weights_only=True)
        assert checkpoint["training"]["objective"] == "dfs_nodes"
        assert checkpoint["training"]["reward_units"] == "nodes_saved"
        assert checkpoint["training"]["cluster_attempts"] == 512
        assert checkpoint["config"]["cluster_attempts"] == 512
        assert checkpoint["cost_weight"] == checkpoint["contradiction_reward"] == 0.
        restored = runtime.load(path)
        assert restored.config["cluster_attempts"] == 512 and restored.config.get("max_total_work") is None
        assert all(torch.isfinite(value).all() for value in restored.state_dict().values())


def test_node_checkpoint_selection_keeps_equal_node_initialization_despite_better_pruning_and_time(tmp_path,
                                                                                               monkeypatch):
    progress = iter((.1, .9))
    calls = []

    def roots(*args, **options):
        assert options["cluster_attempts"] == 512 and options["total_work"] is None
        amount = next(progress)
        return dict(contradictions=0, additional_resolved_fraction=amount,
                    mean_progress=amount, mean_seconds=1 - amount, rows=[])

    original = learn.solve

    def search(network, domains, mode, model, **options):
        if mode == "trained":
            calls.append(options)
            return dict(status="solved", nodes=7, solve_s=1. / len(calls))
        return original(network, domains, mode, model, **options)

    monkeypatch.setattr(learn, "evaluate_roots", roots)
    monkeypatch.setattr(learn, "solve", search)
    output = tmp_path / "ties.pt"
    learn.fit(_data(), output=output, objective="dfs_nodes", epochs=1, width=8, rounds=1,
        rollouts=2, steps=1, warmup=1, teacher_candidates=8, teacher_mutations=0,
        train_indices=[0], validation_indices=[1], validation_seconds=2., search_nodes=100,
        search_seconds=2., max_total_work=None, cluster_attempts=512, max_cores=3,
        max_open_colors=3, max_work=10_000, max_size=256)
    assert len(calls) == 2 and all(options["cluster_attempts"] == 512 for options in calls)
    best = torch.load(output.with_stem("ties.best"), weights_only=True)
    latest = torch.load(output, weights_only=True)
    assert best["history"] == [] and len(latest["history"]) == 1
    assert best["selection"]["search_nodes"] == latest["selection"]["search_nodes"] == 7


def test_all_censored_node_trials_do_not_update_weights_or_imitate_pruning(tmp_path, monkeypatch):
    original_rollout = Policy.rollout

    def force_useful(self, domains, **options):
        if self.training and options.get("trace") is None and not options.get("greedy"):
            options.pop("prefix", None)
            options["trace"] = self.candidate_trace(_candidate(), domains)
        return original_rollout(self, domains, **options)

    def search(network, domains, mode, model, **options):
        return dict(status="solved" if mode == "trained" else "node_limit", nodes=100,
                    setup_s=0., solve_s=0.)

    steps = []
    original_step = torch.optim.Adam.step

    def step(optimizer, *args, **kwargs):
        steps.append(True)
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(Policy, "rollout", force_useful)
    monkeypatch.setattr(learn, "solve", search)
    monkeypatch.setattr(learn.ClusterCandidateGenerator, "generate", lambda *a, **k: [_candidate()])
    monkeypatch.setattr(torch.optim.Adam, "step", step)
    output = tmp_path / "unknown.pt"
    model, history = learn.fit(_data(), output=output, objective="dfs_nodes", epochs=1,
        width=8, rounds=1, rollouts=2, steps=1, warmup=0, teacher_candidates=8,
        teacher_mutations=0, train_indices=[0], validation_indices=[1], validation_seconds=2.,
        search_nodes=100, search_seconds=2., max_cores=3, max_open_colors=3,
        max_work=10_000, max_size=256, curriculum=False)
    assert not steps
    assert history[0].get("updates", 0) == 0 and history[0]["search_limited"] == history[0]["trials"] == 2
    initial = load(output.with_stem("unknown.initial"))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, initial.state_dict()[key], atol=0, rtol=0)


def test_pruning_objective_uses_actual_contraction_without_costs_or_search():
    data = _data()
    _, state, pools, _ = learn.training_states(data, [0], [1])
    with _closed(state, pools[0][0], data["network"]["colors"]):
        before = _snapshot(state)
        plain = learn.measure(_model(data), state, _candidate(), objective="pruning")
        expensive = learn.measure(_model(data), state, _candidate(), objective="pruning",
            cost_weight=1_000., proposal_s=1_000.,
            search=lambda domains: pytest.fail("Pruning labels require no DFS search"))
        assert plain["useful"] == expensive["useful"] == 1
        assert plain["gain"] == expensive["gain"] == 3
        assert plain["reward_valid"] == expensive["reward_valid"] == 1
        assert plain["reward"] == pytest.approx(expensive["reward"])
        assert plain["reward"] > 0
        assert _snapshot(state) == before


@pytest.mark.parametrize("kind", ["skip", "noop", "rejected"])
def test_pruning_objective_completed_noops_are_zero_and_do_not_change_domains(kind):
    data = _data()
    _, state, pools, _ = learn.training_states(data, [0], [1])
    candidate = (None if kind == "skip" else SummaryClusterCandidate("edge", ("ab",), ("a",))
                 if kind == "noop" else _candidate())
    with _closed(state, pools[0][0], data["network"]["colors"]):
        before = _snapshot(state)
        result = learn.measure(_model(data), state, candidate, objective="pruning",
            cost_weight=1_000., proposal_s=1_000.,
            search=lambda domains: pytest.fail("No DFS labels belong to the pruning objective"),
            **(dict(work_budget=1) if kind == "rejected" else {}))
        assert result["reward"] == result["useful"] == result["gain"] == 0
        assert result["reward_valid"] == 1
        assert bool(result["rejected"]) == (kind == "rejected")
        assert _snapshot(state) == before


def test_pruning_actor_warmup_then_reinforce_uses_dfs_only_for_validation(tmp_path, monkeypatch):
    data = _data()
    original = deepcopy(data)
    base, state, pools, validation = learn.training_states(data, [0], [1])
    before = _snapshot(state)
    monkeypatch.setattr(learn, "training_states", lambda *a, **k: (base, state, pools, validation))
    original_measure, original_search = learn.measure, learn.solve
    measured, searches = [], []

    def pruning_measure(model, trial_state, candidate, **options):
        if trial_state is state:
            assert options["objective"] == "pruning" and options.get("search") is None
            options.update(cost_weight=1_000., proposal_s=1_000.)
        result = original_measure(model, trial_state, candidate, **options)
        if trial_state is state:
            assert "search_gain" not in result and 0 <= result["reward"] <= 1
            measured.append(result)
        return result

    def validation_search(network, domains, mode, model, **options):
        assert mode == "trained", "Pruning training must not run counterfactual DFS grading"
        assert all(value.device.type == "cpu" for value in (*model.parameters(), *model.buffers()))
        searches.append(mode)
        return original_search(network, domains, mode, model, **options)

    monkeypatch.setattr(learn, "measure", pruning_measure)
    monkeypatch.setattr(learn, "solve", validation_search)
    output = tmp_path / "pruning_actor.pt"
    model, history = learn.fit(data, output=output, device="cpu", objective="pruning",
        epochs=2, width=8, rounds=1, seed=7, rollouts=2, steps=1, warmup=1,
        teacher_candidates=8, teacher_mutations=0, curriculum=False, checkpoint_every=1,
        train_indices=[0], validation_indices=[1], validation_seconds=5., search_nodes=100,
        cluster_attempts=8, max_cores=3, max_open_colors=3, max_work=10_000, max_size=256,
        target_first=True)
    assert measured and any(row["reward"] > 0 for row in measured)
    assert len(searches) == 3 and len(history) == 2
    assert all(row["updates"] == 1 and row["search_grading"]["calls"] == 0 for row in history)
    assert all(row["teacher_trials"] >= 0 and row["positive_teachers"] > 0 for row in history)
    assert [row["phase"] for row in history] == ["teacher_only", "reinforce_and_imitation"]
    assert all(np.isfinite(row["policy_loss"]) for row in history) and history[1]["prefixed"] == 0
    assert data == original and _snapshot(state) == before and not state._temporary_core_keys
    initial = torch.load(output.with_stem("pruning_actor.initial"), weights_only=True)
    warm = torch.load(output.with_stem("pruning_actor.epoch0001"), weights_only=True)
    latest = torch.load(output, weights_only=True)
    assert latest["training"]["objective"] == "pruning" and latest["training"]["warmup"] == 1
    assert latest["training"]["reward_units"] == "entropy_fraction"
    assert latest["training"]["device"] == latest["training"]["validation_device"] == "cpu"
    assert latest["cost_weight"] == 0. and latest["selection"]["search_complete"] == 1
    assert any(not torch.equal(value, warm["weights"][key]) for key, value in initial["weights"].items())
    assert any(not torch.equal(value, latest["weights"][key]) for key, value in warm["weights"].items())
    restored = runtime.load(output)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)


def test_actor_validation_interval_keeps_skipped_metrics_distinct_and_snapshots_immutable(tmp_path, monkeypatch):
    original_save, original_roots = learn._save, learn.evaluate_roots
    latest, snapshots, validations = [], {}, []
    output = tmp_path / "interval.pt"

    def save(path, *args, **options):
        if ".epoch" in path.stem:
            assert path not in snapshots, "Epoch snapshots must never be overwritten"
        original_save(path, *args, **options)
        if path == output:
            latest.append(torch.load(path, weights_only=True))
        if ".epoch" in path.stem:
            snapshots[path] = path.read_bytes()

    def roots(*args, **options):
        validations.append(True)
        return original_roots(*args, **options)

    monkeypatch.setattr(learn, "_save", save)
    monkeypatch.setattr(learn, "evaluate_roots", roots)
    _, history = learn.fit(_data(), output=output, objective="pruning", epochs=3, width=8,
        rounds=1, seed=7, rollouts=2, steps=1, warmup=1, teacher_candidates=8,
        teacher_mutations=0, train_indices=[0], validation_indices=[1], validation_seconds=5.,
        max_cores=3, max_open_colors=3, max_work=10_000, max_size=256,
        validation_interval=2, checkpoint_every=1)
    assert len(validations) == 3  # Initialization, scheduled epoch 2, and final epoch 3.
    assert len(latest) == 3 and [len(checkpoint["history"]) for checkpoint in latest] == [1, 2, 3]
    assert history[0]["validation"] is None
    assert latest[0]["selection"] == dict(source="skipped", epoch=1, last_validation_epoch=0)
    assert all(row["validation"]["source"] == "validation_roots" for row in history[1:])
    assert len(snapshots) == 3 and all(path.read_bytes() == contents for path, contents in snapshots.items())
    assert torch.load(output.with_stem("interval.initial"), weights_only=True)["history"] == []
    best = torch.load(output.with_stem("interval.best"), weights_only=True)
    assert len(best["history"]) in {0, 2, 3}  # A skipped epoch cannot select the best checkpoint.


@pytest.mark.parametrize("options", [dict(validation_interval=0), dict(validation_interval=True),
    dict(validation_interval=1.5), dict(checkpoint_every=-1), dict(checkpoint_every=True),
    dict(checkpoint_every=1.5), dict(device="meta"), dict(puzzles_per_epoch=-1),
    dict(puzzles_per_epoch=True), dict(puzzles_per_epoch=1.5), dict(progress_every=0),
    dict(progress_every=True), dict(teacher_cache_size=-1), dict(teacher_cache_size=True),
    dict(cycle_puzzles=1)])
def test_actor_invalid_device_or_schedule_fails_before_creating_outputs(tmp_path, options):
    with pytest.raises(ValueError):
        learn.fit(_data(), output=tmp_path / "new_run" / "policy.pt", **options)
    assert not list(tmp_path.iterdir())


def test_actor_cuda_unavailable_fails_before_creating_states_or_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(learn, "training_states", lambda *a, **k: pytest.fail("CUDA must be checked first"))
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        learn.fit(_data(), output=tmp_path / "new_run" / "policy.pt", device="cuda")
    assert not list(tmp_path.iterdir())


def test_actor_existing_epoch_snapshot_blocks_run_without_touching_files(tmp_path):
    output = tmp_path / "policy.pt"
    collision = output.with_stem("policy.epoch0002")
    collision.write_bytes(b"preserve this checkpoint")
    with pytest.raises(FileExistsError, match="already exists"):
        learn.fit(_data(), output=output, epochs=3, checkpoint_every=2)
    assert list(tmp_path.iterdir()) == [collision] and collision.read_bytes() == b"preserve this checkpoint"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA-enabled PyTorch environment and GPU are required")
def test_actor_cuda_training_has_gpu_gradients_cpu_validation_and_portable_checkpoint(tmp_path, monkeypatch):
    data = _data()
    base, state, pools, validation = learn.training_states(data, [0], [1])
    before = _snapshot(state)
    monkeypatch.setattr(learn, "training_states", lambda *a, **k: (base, state, pools, validation))
    original_roots, original_search, original_step = learn.evaluate_roots, learn.solve, torch.optim.Adam.step
    calls = dict(step=0, propagation=0, search=0)

    def gpu_step(optimizer, *args, **options):
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        assert any(parameter.grad is not None for parameter in parameters)
        assert all(parameter.device.type == "cuda" and
                   (parameter.grad is None or parameter.grad.device.type == "cuda") for parameter in parameters)
        result = original_step(optimizer, *args, **options)
        assert all(item["exp_avg"].device.type == "cuda" for item in optimizer.state.values())
        calls["step"] += 1
        return result

    def cpu_roots(network, domains, model, **options):
        assert all(value.device.type == "cpu" for value in (*model.parameters(), *model.buffers()))
        calls["propagation"] += 1
        return original_roots(network, domains, model, **options)

    def cpu_search(network, domains, mode, model, **options):
        assert mode == "trained"
        assert all(value.device.type == "cpu" for value in (*model.parameters(), *model.buffers()))
        calls["search"] += 1
        return original_search(network, domains, mode, model, **options)

    monkeypatch.setattr(torch.optim.Adam, "step", gpu_step)
    monkeypatch.setattr(learn, "evaluate_roots", cpu_roots)
    monkeypatch.setattr(learn, "solve", cpu_search)
    output = tmp_path / "cuda_actor.pt"
    model, history = learn.fit(data, output=output, device="cuda", objective="pruning",
        epochs=2, width=8, rounds=1, seed=7, rollouts=2, steps=1, warmup=1,
        teacher_candidates=8, teacher_mutations=0, train_indices=[0], validation_indices=[1],
        validation_seconds=10., search_nodes=100, cluster_attempts=8, max_cores=3,
        max_open_colors=3, max_work=10_000, max_size=256, target_first=True, curriculum=False)
    assert calls["step"] == sum(row["updates"] for row in history) == 2
    assert calls["propagation"] == calls["search"] == 3
    assert all(parameter.device.type == "cuda" for parameter in model.parameters())
    assert _snapshot(state) == before and not state._temporary_core_keys
    checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    assert checkpoint["training"]["device"] == "cuda" and checkpoint["training"]["validation_device"] == "cpu"
    restored = runtime.load(output)
    assert all(parameter.device.type == "cpu" for parameter in restored.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), restored.state_dict()[key], atol=0, rtol=0)


class _ReplayQ(torch.nn.Module):
    """Different raw values for finishing a prefix and extending that prefix."""
    def __init__(self, values=None):
        super().__init__()
        self.network = {"colors": ["a"]}
        self.values = torch.nn.Parameter(torch.tensor(values or [0.] * 6))

    def encode(self, domains):
        return None

    def q_values(self, encoded, sizes, keys, opened, output_phase):
        if not opened:
            actions, indices = [("open", 0)], [0]
        elif not keys:
            actions, indices = [("core", 0)], [1]
        elif output_phase:
            actions, indices = [("finish", -1)], [4 if len(keys) == 1 else 5]
        elif len(keys) == 1:
            actions, indices = [("finish", -1), ("core", 1)], [2, 3]
        else:
            actions, indices = [("finish", -1)], [5]
        return actions, self.values[indices]


def _fixed_q_replay(monkeypatch, *, reward=1., noop=False, offset=2):
    from experiments.cluster_selection.reinforcement import qlearn
    replay = qlearn.Replay(capacity=8, n_step=8)
    trace = [("open", 0), ("core", 0)]
    if not noop:
        trace.append(("core", 1))
    trace += [("finish", -1), ("finish", -1)]
    replay.add(0, [(0, 1)], trace, reward)
    episode = replay.episodes[-1]
    monkeypatch.setattr(replay, "sample", lambda rng, count: [(episode, offset)] * count)
    return replay


def test_q_noop_finish_does_not_label_a_useful_shared_prefix_extension_zero(monkeypatch):
    from experiments.cluster_selection.reinforcement import qlearn
    noop = _fixed_q_replay(monkeypatch, reward=0., noop=True)
    useful = _fixed_q_replay(monkeypatch, reward=1.)
    student, target = _ReplayQ([0., 0., .5, .5, 0., 0.]), _ReplayQ()
    qlearn.q_loss(student, target, noop, 1, Random(0), n_step=8).backward()
    assert student.values.grad[2] > 0  # Decrease the action that finishes the no-op.
    assert student.values.grad[3] == 0
    student.zero_grad(set_to_none=True)
    qlearn.q_loss(student, target, useful, 1, Random(0), n_step=8).backward()
    assert student.values.grad[3] < 0  # Increase the extension leading to pruning.
    assert student.values.grad[2] == 0
    assert target.values.grad is None


def test_q_measured_terminal_return_reaches_early_actions_across_a_long_horizon(monkeypatch):
    from experiments.cluster_selection.reinforcement import qlearn
    replay = _fixed_q_replay(monkeypatch, reward=.75, offset=0)
    student, target = _ReplayQ(), _ReplayQ()
    loss = qlearn.q_loss(student, target, replay, 1, Random(1), n_step=8)
    assert loss.item() == pytest.approx(.5 * .75 ** 2)
    loss.backward()
    assert student.values.grad[0] == pytest.approx(-.75)
    assert torch.count_nonzero(student.values.grad).item() == 1
    assert target.values.grad is None


def test_q_actual_terminal_action_has_no_bootstrap(monkeypatch):
    from experiments.cluster_selection.reinforcement import qlearn
    replay = _fixed_q_replay(monkeypatch, reward=.75, offset=4)
    student, target = _ReplayQ(), _ReplayQ([1_000.] * 6)
    loss = qlearn.q_loss(student, target, replay, 1, Random(1), n_step=8)
    assert loss.item() == pytest.approx(.5 * .75 ** 2)
    loss.backward()
    assert student.values.grad[5] == pytest.approx(-.75)
    assert target.values.grad is None


def test_q_weaker_measured_completion_cannot_erase_a_better_prefix_continuation(monkeypatch):
    from experiments.cluster_selection.reinforcement import qlearn
    replay = _fixed_q_replay(monkeypatch, reward=.2, offset=0)
    student, target = _ReplayQ(), _ReplayQ([0., .9, 0., 0., 0., 0.])
    loss = qlearn.q_loss(student, target, replay, 1, Random(1), n_step=8)
    assert loss.item() == pytest.approx(.5 * .9 ** 2)
    loss.backward()
    assert student.values.grad[0] == pytest.approx(-.9)
    assert target.values.grad is None


def test_q_zero_terminal_rollout_preserves_a_promising_unfinished_prefix(monkeypatch):
    from experiments.cluster_selection.reinforcement import qlearn
    replay = _fixed_q_replay(monkeypatch, reward=0., noop=True, offset=0)
    student, target = _ReplayQ(), _ReplayQ([0., .6, 0., 0., 0., 0.])
    loss = qlearn.q_loss(student, target, replay, 1, Random(1), n_step=8)
    assert loss.item() == pytest.approx(.5 * .6 ** 2)
    loss.backward()
    assert student.values.grad[0] == pytest.approx(-.6)
    assert target.values.grad is None


def test_q_measured_positive_completion_is_a_lower_bound_at_its_own_prefix(monkeypatch):
    from experiments.cluster_selection.reinforcement import qlearn
    replay = _fixed_q_replay(monkeypatch, reward=.75, offset=0)
    student, target = _ReplayQ(), _ReplayQ([0., .1, 0., 0., 0., 0.])
    loss = qlearn.q_loss(student, target, replay, 1, Random(1), n_step=1)
    assert loss.item() == pytest.approx(.5 * .75 ** 2)
    loss.backward()
    assert student.values.grad[0] == pytest.approx(-.75)
    assert target.values.grad is None


def test_q_nonterminal_backup_uses_raw_legal_target_values_and_detaches_them(monkeypatch):
    from experiments.cluster_selection.reinforcement import qlearn
    replay = _fixed_q_replay(monkeypatch, reward=.75, offset=2)
    # After the extension, only finish is legal and its raw value is .8.
    # Large values at other prefixes and probability normalization cannot enter.
    student, target = _ReplayQ(), _ReplayQ([1_000.] * 5 + [.8])
    loss = qlearn.q_loss(student, target, replay, 1, Random(1), n_step=1)
    assert loss.item() == pytest.approx(.5 * .8 ** 2)
    loss.backward()
    assert student.values.grad[3] == pytest.approx(-.8)
    assert torch.count_nonzero(student.values.grad).item() == 1
    assert target.values.grad is None


def test_q_replay_keeps_noops_snapshots_and_puzzle_balanced_sampling():
    from experiments.cluster_selection.reinforcement import qlearn
    replay = qlearn.Replay(capacity=16, n_step=8)
    domains, trace = [[0, 1]], [("open", 0), ("core", 0), ("finish", -1), ("finish", -1)]
    replay.add(0, domains, trace, 0., source="student", removed=0)
    domains[0].clear()
    trace.clear()
    first = replay.episodes[0]
    assert first["domains"] == ((0, 1),)
    assert len(first["trace"]) == 4 and first["reward"] == 0.
    assert first["source"] == "student" and first["removed"] == 0
    for idx in range(1, 8):
        replay.add(0, [(idx, idx + 1)], first["trace"], 1., source="teacher")
    replay.add(1, [(0, 1)], first["trace"], 0., source="student")
    draws = replay.sample(Random(3), 1_000)
    assert .4 < sum(episode["puzzle"] == 1 for episode, _ in draws) / len(draws) < .6
    assert all(0 <= offset < len(episode["trace"]) for episode, offset in draws)
    bounded = qlearn.Replay(capacity=3, n_step=8)
    for idx in range(8):
        bounded.add(idx, [(0, 1)], first["trace"], float(idx % 2))
    assert len(bounded.episodes) == 3


def _q_data():
    closing = np.ones((4, 4), dtype=bool)
    closing[1, 1] = closing[3, 3] = False
    cores = {key: NumpyCore(values=np.eye(4, dtype=bool), colors=list(key))
             for key in ("ab", "bc", "ac")}
    cores["tight"] = NumpyCore(values=closing, colors=["a", "c"])
    raw = network_data(ConstraintNetwork(cores, maintain_indices=True))
    core_indices = {core["key"]: idx for idx, core in enumerate(raw["cores"])}
    initial = [{color: list(values) for color in raw["colors"]} for values in ((0, 1), (2, 3))]
    candidates = [dict(cores=[core_indices[key] for key in keys], open=[raw["colors"].index("a")],
                       gain=1_000_000., contradiction=False, time=0., estimate=[0, 0, 0])
                  for keys in (("ab", "bc", "ac"), ("ab", "bc", "tight"))]
    return dict(version=1, network=raw, reward_colors=raw["colors"], initial_domains=initial,
        train_indices=[0], validation_indices=[1],
        samples=[dict(puzzle=0, domains=[initial[0][color] for color in raw["colors"]],
                      candidates=candidates)])


@pytest.mark.parametrize("method", ["actor", "q"])
@pytest.mark.parametrize("nested,maximum,count,expected", [
    (False, None, 8, 8192), (True, None, 50, 16384), (True, 65536, 256, 65536)])
def test_training_records_and_uses_shared_structural_search(tmp_path, monkeypatch, capsys,
                                                          method, nested, maximum, count, expected):
    from experiments.cluster_selection.candidate_generation import ClusterCandidateGenerator
    from experiments.cluster_selection.reinforcement import qlearn
    learner = learn if method == "actor" else qlearn
    data = _data() if method == "actor" else _q_data()
    original_data = deepcopy(data)
    calls, generate = [], ClusterCandidateGenerator.generate_nested

    def observed(generator, *args, **kwargs):
        calls.append((generator.max_candidates, generator.max_steps))
        return generate(generator, *args, **kwargs)

    monkeypatch.setattr(ClusterCandidateGenerator, "generate_nested", observed)
    output = tmp_path / "policy.pt"
    options = dict(epochs=2, width=8, rounds=1, seed=7, puzzles_per_epoch=1,
        steps=1, rollouts=2, teacher_candidates=count, teacher_mutations=0, teacher_probes=0,
        teacher_max_steps=maximum, nested_teacher=nested, train_indices=[0], validation_indices=[1],
        validation_seconds=5., search_nodes=100, cluster_attempts=8, checkpoint_every=1,
        max_cores=3, max_open_colors=3, max_work=10_000, max_size=256)
    options.update(dict(objective="pruning", warmup=1, curriculum=False, target_first=True,
                        max_total_work=None) if method == "actor" else dict(batch_size=2, updates_per_state=1))
    _, history = learner.fit(data, output=output, **options)
    assert bool(calls) == nested
    assert all(item == (count, expected) for item in calls)
    assert data == original_data
    assert "structural proposals" in capsys.readouterr().out
    assert sum(row["teacher_structural_calls"] for row in history) > 0
    assert sum(row["teacher_structural_candidates"] for row in history) > 0
    assert sum(row["teacher_structural_useful"] for row in history) > 0
    for path in (output, output.with_stem("policy.initial"), output.with_stem("policy.best"),
                 output.with_stem("policy.epoch0001"), output.with_stem("policy.epoch0002")):
        saved = torch.load(path, weights_only=True)
        assert saved["training"]["teacher_candidates"] == count
        assert saved["training"]["teacher_max_steps"] == expected
        assert saved["training"]["nested_teacher"] is nested
        assert "teacher_max_steps" not in saved["config"] and "nested_teacher" not in saved["config"]


@pytest.mark.parametrize("method", ["actor", "q"])
@pytest.mark.parametrize("options", [dict(teacher_max_steps=0), dict(teacher_max_steps=True),
    dict(teacher_max_steps=1.5), dict(nested_teacher=1), dict(teacher_candidates=True)])
def test_bad_teacher_search_fails_before_writing(tmp_path, method, options):
    from experiments.cluster_selection.reinforcement import qlearn
    learner = learn if method == "actor" else qlearn
    with pytest.raises(ValueError):
        learner.fit(_q_data(), output=tmp_path / "policy.pt", **options)
    assert not list(tmp_path.iterdir())


def test_nested_q_teacher_allows_zero_structural_candidates(tmp_path):
    from experiments.cluster_selection.reinforcement import qlearn
    output = tmp_path / "policy.pt"
    _, history = qlearn.fit(_q_data(), output=output, epochs=1, width=8, rounds=1, seed=7,
        steps=1, rollouts=2, teacher_candidates=0, teacher_mutations=0, teacher_probes=0,
        nested_teacher=True, batch_size=2, updates_per_state=1, max_cores=3,
        max_work=10_000, max_size=256, cluster_attempts=4, validation_seconds=5.)
    assert history[0]["teacher_structural_candidates"] == 0
    assert torch.load(output, weights_only=True)["training"]["teacher_max_steps"] == 8192


@pytest.mark.parametrize("options", [dict(replay_capacity=0), dict(n_step=0),
    dict(target_update=0), dict(epsilon_start=-.1), dict(epsilon_end=1.1),
    dict(validation_interval=0), dict(validation_interval=True), dict(validation_interval=1.5),
    dict(checkpoint_every=-1), dict(checkpoint_every=True), dict(checkpoint_every=1.5),
    dict(puzzles_per_epoch=-1), dict(puzzles_per_epoch=True), dict(puzzles_per_epoch=1.5),
    dict(progress_every=0), dict(progress_every=True), dict(cycle_puzzles=1)])
def test_q_learning_rejects_invalid_parameters_before_writing(tmp_path, options):
    from experiments.cluster_selection.reinforcement import qlearn
    with pytest.raises(ValueError):
        qlearn.fit(_q_data(), output=tmp_path / "invalid.pt", **options)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("method", ["actor", "q"])
@pytest.mark.parametrize("puzzles_per_epoch", [0, 2, 5])
@pytest.mark.parametrize("cycle_puzzles", [False, True])
def test_training_full_pass_and_capped_exposure_count_real_puzzle_visits(tmp_path, monkeypatch, capsys,
                                                                       method, puzzles_per_epoch, cycle_puzzles):
    from experiments.cluster_selection.reinforcement import qlearn
    learner = learn if method == "actor" else qlearn
    data = _data() if method == "actor" else _q_data()
    colors = data["network"]["colors"]
    data.update(initial_domains=[{color: list(values) for color in colors}
                                for values in ((0, 1), (0, 2), (0, 3), (2, 3))],
                train_indices=[0, 1, 2], validation_indices=[3])
    data.pop("samples")
    base, state, pools, validation = learn.training_states(data)
    assert set(pools) == {0, 1, 2}
    monkeypatch.setattr(learner, "training_states", lambda *a, **k: (base, state, pools, validation))
    visited, original_closed = [], learner._closed

    def closed(current, sample, ordered_colors):
        visited.append(next(puzzle for puzzle, entries in pools.items()
                            if any(item is sample for item in entries)))
        return original_closed(current, sample, ordered_colors)

    monkeypatch.setattr(learner, "_closed", closed)
    epochs = 3 if cycle_puzzles else 2
    options = dict(epochs=epochs, width=8, rounds=1, seed=7, steps=1, puzzles_per_epoch=puzzles_per_epoch,
        teacher_candidates=8, teacher_mutations=0, validation_seconds=5., max_cores=3,
        max_open_colors=1, max_work=10_000, max_size=256, progress_every=1, validation_interval=2,
        cycle_puzzles=cycle_puzzles)
    if method == "actor":
        options.update(objective="pruning", rollouts=2, warmup=1, teacher_cache_size=0, curriculum=False)
    else:
        options.update(rollouts=1, batch_size=2, updates_per_state=1, target_update=2,
                       replay_capacity=64, search_nodes=100, cluster_attempts=8)
    output = tmp_path / f"{method}.pt"
    _, history = learner.fit(data, output=output, **options)
    count = min(puzzles_per_epoch or len(pools), len(pools))
    counts = [count, 1, count] if cycle_puzzles and count == 2 else [count] * epochs
    cumulative = np.cumsum(counts).tolist()
    assert len(visited) == sum(counts)
    offset = 0
    for actual_count in counts:
        assert len(set(visited[offset:offset + actual_count])) == actual_count
        if puzzles_per_epoch in (0, 5):
            assert set(visited[offset:offset + actual_count]) == {0, 1, 2}
        offset += actual_count
    if cycle_puzzles:
        assert len(set(visited[:3])) == 3  # The final short chunk completes the pass before any repetition.
    assert [row["puzzle_visits"] for row in history] == counts
    assert [row["total_puzzle_visits"] for row in history] == cumulative
    assert [row["total_state_exposures"] for row in history] == cumulative
    assert [row["puzzle_passes"] for row in history] == pytest.approx([total / 3 for total in cumulative])
    saved = torch.load(output, weights_only=True)["training"]
    assert saved["puzzles_per_epoch"] == puzzles_per_epoch
    assert saved["puzzles_per_epoch_effective"] == count
    assert saved["cycle_puzzles"] is cycle_puzzles
    assert saved["planned_puzzle_visits"] == saved["puzzle_visits"] == sum(counts)
    assert saved["planned_puzzle_passes"] == pytest.approx(sum(counts) / 3)
    assert saved["state_exposures"] == sum(counts)
    assert saved["unique_puzzles_visited"] == len(set(visited))
    assert saved["train_puzzle_count"] == saved["train_context_count_initial"] == 3
    assert saved["dataset_instances"] == 4 and saved["validation_instances"] == 1
    assert saved["train_context_count"] == sum(map(len, pools.values()))
    expected_updates = sum(row["updates"] if method == "actor" else row["epoch_updates"] for row in history)
    assert saved["updates"] == expected_updates
    initial = torch.load(output.with_stem(output.stem + ".initial"), weights_only=True)["training"]
    assert initial["puzzle_visits"] == initial["state_exposures"] == initial["updates"] == 0
    log = capsys.readouterr().out
    assert "3 unresolved training puzzles" in log and "puzzles 1/" in log
    unit = "chunk" if cycle_puzzles and count < 3 else "epoch"
    assert saved["epoch_unit"] == unit and f"{unit} 1/{epochs}" in log
    if unit == "chunk":
        assert "2 chunks per full pass" in log


@pytest.mark.parametrize("teacher_cache_size", [0, 1])
def test_teacher_cache_can_retain_all_measured_states_across_dataset_passes(tmp_path, monkeypatch,
                                                                         teacher_cache_size):
    calls = []

    def generate(self, domains, **options):
        calls.append(tuple(domains.values()))
        return ()

    def skip(self, domains, **options):
        zero = self.head[-1].weight.sum() * 0
        return None, zero, zero, []

    monkeypatch.setattr(learn.ClusterCandidateGenerator, "generate", generate)
    monkeypatch.setattr(Policy, "rollout", skip)
    output = tmp_path / "cached.pt"
    _, history = learn.fit(_data(), output=output, epochs=2, width=8, rounds=1, seed=7,
        train_indices=[0, 1], validation_indices=[], puzzles_per_epoch=0, rollouts=2,
        steps=1, warmup=2, teacher_candidates=8, teacher_mutations=0,
        teacher_cache_size=teacher_cache_size, objective="pruning", curriculum=False)
    assert len(calls) == 2 if teacher_cache_size == 0 else len(calls) > 2
    assert len(set(calls)) == 2
    assert all(row["updates"] == 0 and row["puzzle_visits"] == 2 for row in history)
    saved = torch.load(output, weights_only=True)["training"]
    assert saved["teacher_cache_size"] == teacher_cache_size and saved["updates"] == 0


@pytest.mark.parametrize("existing_snapshot", [False, True])
def test_q_learning_never_overwrites_an_existing_run(tmp_path, existing_snapshot):
    from experiments.cluster_selection.reinforcement import qlearn
    output = tmp_path / "existing.pt"
    existing = output.with_stem("existing.epoch0002") if existing_snapshot else output
    existing.write_bytes(b"preserve this experiment")
    with pytest.raises(FileExistsError):
        qlearn.fit(_q_data(), output=output, epochs=3, checkpoint_every=2)
    assert list(tmp_path.iterdir()) == [existing] and existing.read_bytes() == b"preserve this experiment"


def test_q_validation_interval_saves_current_weights_without_stale_metrics(tmp_path, monkeypatch):
    from experiments.cluster_selection.reinforcement import qlearn
    original_save, original_roots = qlearn._save, qlearn.evaluate_roots
    latest, snapshots, validations = [], {}, []
    output = tmp_path / "q_interval.pt"

    def save(path, *args, **options):
        if ".epoch" in path.stem:
            assert path not in snapshots, "Epoch snapshots must never be overwritten"
        original_save(path, *args, **options)
        if path == output:
            latest.append(torch.load(path, weights_only=True))
        if ".epoch" in path.stem:
            snapshots[path] = path.read_bytes()

    def roots(*args, **options):
        validations.append(True)
        return original_roots(*args, **options)

    monkeypatch.setattr(qlearn, "_save", save)
    monkeypatch.setattr(qlearn, "evaluate_roots", roots)
    model, history = qlearn.fit(_q_data(), output=output, epochs=3, width=8, rounds=1,
        lr=.003, seed=7, puzzles_per_epoch=1, rollouts=1, steps=1, teacher_candidates=0,
        teacher_mutations=0, batch_size=2, updates_per_state=1, target_update=2,
        replay_capacity=32, validation_seconds=5., search_nodes=100, cluster_attempts=8,
        max_cores=3, max_open_colors=1, max_work=10_000, max_size=256, value_encoding=True,
        validation_interval=2, checkpoint_every=1)
    assert len(validations) == 3 and len(latest) == 3
    assert [len(checkpoint["history"]) for checkpoint in latest] == [1, 2, 3]
    assert history[0]["validation"] is None
    assert latest[0]["selection"] == dict(source="skipped", epoch=1, last_validation_epoch=0)
    assert all(row["validation"]["search_complete"] == 1 for row in history[1:])
    assert latest[-1]["training"]["last_validation_epoch"] == 3
    assert all(path.read_bytes() == contents for path, contents in snapshots.items()) and len(snapshots) == 3
    assert all(checkpoint["replay"]["episodes"] and checkpoint["optimizer"]["state"] for checkpoint in latest)
    best = torch.load(output.with_stem("q_interval.best"), weights_only=True)
    assert len(best["history"]) in {0, 2, 3}
    restored = runtime.load(output)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)


def test_q_learning_regrades_saved_trials_keeps_failures_and_roundtrips_a_real_policy(tmp_path, monkeypatch):
    from experiments.cluster_selection.reinforcement import qlearn
    data = _q_data()
    original = deepcopy(data)
    base, state, pools, validation = qlearn.training_states(data)
    before = _snapshot(state)
    monkeypatch.setattr(qlearn, "training_states", lambda *args: (base, state, pools, validation))
    output = tmp_path / "q.pt"
    model, history = qlearn.fit(data, output=output, epochs=3, width=8, rounds=1,
        lr=.003, seed=7, puzzles_per_epoch=1, rollouts=2, steps=1, teacher_candidates=0,
        teacher_mutations=0, teacher_probes=0, batch_size=16, updates_per_state=16,
        n_step=8, target_update=8, replay_capacity=128, epsilon_start=.3, epsilon_end=.1,
        validation_seconds=5., search_nodes=100, cluster_attempts=8, max_total_work=None,
        max_cores=3, max_open_colors=1, max_work=10_000, max_size=256, value_encoding=True,
        device="cpu")
    assert data == original and _snapshot(state) == before and not state._temporary_core_keys
    assert len(history) == 3 and history[-1]["updates"] > 0
    checkpoint = torch.load(output, weights_only=True)
    episodes = checkpoint["replay"]["episodes"]
    saved = [episode for episode in episodes if episode["source"] == "saved"]
    assert any(episode["reward"] == 0 for episode in saved)
    assert any(0 < episode["reward"] <= 1 for episode in saved)
    assert all(0 <= episode["reward"] <= 1 for episode in episodes)
    assert all(len(episode["domains"]) == len(episode["domains_after"]) == 3 for episode in episodes)
    assert checkpoint["training"]["objective"] == "pruning"
    assert checkpoint["training"]["gamma"] == 1.
    assert checkpoint["training"]["device"] == checkpoint["training"]["validation_device"] == "cpu"
    assert checkpoint["optimizer"] and checkpoint["target_weights"]
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert all(value.device.type == "cpu" for value in checkpoint["target_weights"].values())
    assert all(value.device.type == "cpu" for item in checkpoint["optimizer"]["state"].values()
               for value in item.values() if isinstance(value, torch.Tensor))
    restored = runtime.load(output)
    assert restored.config["q_learning"] and restored.config["cluster_attempts"] == 8
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)
    initial = torch.load(output.with_stem("q.initial"), weights_only=True)
    assert initial["selection"]["greedy_useful"] == 0
    assert checkpoint["selection"]["greedy_reward"] > initial["selection"]["greedy_reward"]
    assert checkpoint["selection"]["greedy_useful"] == 1
    assert checkpoint["selection"]["search_complete"] == 1
    assert checkpoint["selection"]["search_nodes"] == 1
    best = torch.load(output.with_stem("q.best"), weights_only=True)
    assert best["history"] == []  # More pruning cannot break an equal-node validation tie.
    # Reuse rewards only under the same data, representation, and grading units.
    for change in (None, "reward_colors", "data_hash", "reward_units", "config"):
        parent = output
        if change not in (None, "config"):
            altered = deepcopy(checkpoint)
            if change == "reward_colors":
                altered["reward_colors"] = ["a"]
            elif change == "data_hash":
                altered["training"]["data_sha256"] = "another dataset"
            else:
                altered["training"]["reward_units"] = "nodes_saved"
            parent = tmp_path / f"source_{change}.pt"
            torch.save(altered, parent)
        warm_output = tmp_path / f"warm_{change}.pt"
        qlearn.fit(data, output=warm_output, initial_checkpoint=parent, epochs=1,
            width=8, rounds=1, lr=.003, seed=7, puzzles_per_epoch=1, rollouts=1, steps=1,
            teacher_candidates=0, teacher_mutations=0, teacher_probes=0, batch_size=2,
            updates_per_state=1, n_step=8, target_update=8, replay_capacity=128,
            epsilon_start=.3, epsilon_end=.1, validation_seconds=5., search_nodes=100,
            cluster_attempts=8, max_total_work=None, max_cores=4 if change == "config" else 3,
            max_open_colors=1, max_work=10_000, max_size=256, value_encoding=True)
        warm_initial = torch.load(warm_output.with_stem(warm_output.stem + ".initial"), weights_only=True)
        expected = len(episodes) if change is None else 0
        assert warm_initial["training"]["replay_reused"] == expected
        assert len(warm_initial["replay"]["episodes"]) == expected
        assert _snapshot(state) == before and data == original


def test_q_cuda_request_fails_before_creating_states_or_outputs_when_unavailable(tmp_path, monkeypatch):
    from experiments.cluster_selection.reinforcement import qlearn
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(qlearn, "training_states", lambda *args: pytest.fail("CUDA must be checked first"))
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        qlearn.fit(_q_data(), output=tmp_path / "new_run" / "policy.pt", device="cuda")
    assert not list(tmp_path.iterdir())


def test_q_unsupported_training_device_fails_before_writing(tmp_path):
    from experiments.cluster_selection.reinforcement import qlearn
    with pytest.raises(ValueError, match="CPU or CUDA"):
        qlearn.fit(_q_data(), output=tmp_path / "invalid.pt", device="meta")
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA-enabled PyTorch environment and GPU are required")
def test_q_cuda_training_has_real_gpu_gradients_cpu_validation_and_portable_checkpoint(tmp_path, monkeypatch):
    from experiments.cluster_selection.reinforcement import qlearn
    data = _q_data()
    base, state, pools, validation = qlearn.training_states(data)
    before = _snapshot(state)
    monkeypatch.setattr(qlearn, "training_states", lambda *args: (base, state, pools, validation))
    original_loss, original_roots, original_search = qlearn.q_loss, qlearn.evaluate_roots, qlearn.solve
    original_step = torch.optim.Adam.step
    calls = dict(loss=0, step=0, propagation=0, search=0)

    def gpu_loss(model, target, *args, **options):
        assert all(parameter.device.type == "cuda" for parameter in model.parameters())
        assert all(parameter.device.type == "cuda" and not parameter.requires_grad
                   for parameter in target.parameters())
        calls["loss"] += 1
        loss = original_loss(model, target, *args, **options)
        assert loss.device.type == "cuda" and loss.requires_grad
        return loss

    def gpu_step(optimizer, *args, **options):
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        assert any(parameter.grad is not None for parameter in parameters)
        assert all(parameter.device.type == "cuda" and
                   (parameter.grad is None or parameter.grad.device.type == "cuda") for parameter in parameters)
        calls["step"] += 1
        result = original_step(optimizer, *args, **options)
        assert all(item["exp_avg"].device.type == "cuda" for item in optimizer.state.values())
        return result

    def cpu_roots(network, domains, model, **options):
        assert all(value.device.type == "cpu" for value in (*model.parameters(), *model.buffers()))
        calls["propagation"] += 1
        return original_roots(network, domains, model, **options)

    def cpu_search(network, domains, mode, model, **options):
        assert all(value.device.type == "cpu" for value in (*model.parameters(), *model.buffers()))
        calls["search"] += 1
        return original_search(network, domains, mode, model, **options)

    monkeypatch.setattr(qlearn, "q_loss", gpu_loss)
    monkeypatch.setattr(torch.optim.Adam, "step", gpu_step)
    monkeypatch.setattr(qlearn, "evaluate_roots", cpu_roots)
    monkeypatch.setattr(qlearn, "solve", cpu_search)
    output = tmp_path / "cuda.pt"
    model, history = qlearn.fit(data, output=output, device="cuda", epochs=1, width=8, rounds=1,
        lr=.003, seed=7, puzzles_per_epoch=1, rollouts=1, steps=1, teacher_candidates=0,
        teacher_mutations=0, batch_size=2, updates_per_state=1, n_step=8, target_update=1,
        replay_capacity=16, validation_seconds=10., search_nodes=100, cluster_attempts=8,
        max_cores=3, max_open_colors=1, max_work=10_000, max_size=256, value_encoding=True)
    assert calls["loss"] == calls["step"] == history[0]["updates"] == 2
    assert calls["propagation"] == calls["search"] == 2
    assert all(parameter.device.type == "cuda" for parameter in model.parameters())
    assert _snapshot(state) == before and not state._temporary_core_keys
    checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    assert checkpoint["training"]["device"] == "cuda" and checkpoint["training"]["validation_device"] == "cpu"
    assert checkpoint["replay"]["episodes"] and checkpoint["optimizer"]["state"]
    restored = runtime.load(output)
    assert all(parameter.device.type == "cpu" for parameter in restored.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), restored.state_dict()[key], atol=0, rtol=0)
