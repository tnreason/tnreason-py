"""Learned construction must see partial topology and retain a free action space."""
import numpy as np
import pytest
from math import log1p
from random import Random
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from experiments.cluster_selection.candidate_evaluation_mac import evaluate_candidate
from experiments.cluster_selection.data import network_data
from experiments.cluster_selection import runtime

torch = pytest.importorskip("torch")
from experiments.cluster_selection.reinforcement.model import Policy


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _k5():
    # Regular axis orientation makes the original full-network embeddings
    # identical, so only the selected topology can distinguish the extensions.
    cores = {f"{i}{(i + delta) % 5}": NumpyCore(values=~np.eye(2, dtype=bool),
             colors=[str(i), str((i + delta) % 5)]) for i in range(5) for delta in (1, 2)}
    return ConstraintNetwork(cores, maintain_indices=True)


def _model(**options):
    return Policy(network_data(_k5()), width=8, rounds=3, heuristic_candidates=0, **options)


def test_legacy_defaults_preserve_config_weights_and_scores():
    torch.manual_seed(13)
    original = _model()
    torch.manual_seed(13)
    explicit = _model(contextual=False)
    assert original.config == explicit.config and "contextual" not in original.config
    assert not any(key.startswith("selection_") for key in original.state_dict())
    domains = [[0, 1]] * 5
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], atol=0, rtol=0)
    left = original.rollout(domains, beta=0, greedy=True)
    right = explicit.rollout(domains, beta=0, greedy=True)
    assert left[0] == right[0] and left[3] == right[3]
    torch.testing.assert_close(left[1], right[1], atol=0, rtol=0)


def _proposal_model(**options):
    network = ConstraintNetwork({key: NumpyCore(values=np.ones((2, 2), dtype=bool), colors=list(key))
                                 for key in ("ab", "bc")}, maintain_indices=True)
    return Policy(network_data(network), width=8, rounds=1, max_cores=2,
                  max_open_colors=2, heuristic_candidates=0, **options).eval()


@pytest.mark.parametrize("hierarchical", [False, True])
def test_proposals_preserve_first_greedy_choice_rng_and_grad_context(monkeypatch, hierarchical):
    torch.manual_seed(13)
    model = _proposal_model(contextual=True, target_first=True, hierarchical_greedy=hierarchical)
    model.beta = 0
    domains = dict(a=(0, 1), b=(0, 1), c=(0, 1))
    expected, before = model.propose(domains), torch.get_rng_state().clone()
    assert expected is not None
    encode, distribution, calls = model.encode, model.distribution, dict(encode=0, distribution=0)

    def checked_encode(*args, **kwargs):
        assert not torch.is_grad_enabled()
        calls["encode"] += 1
        return encode(*args, **kwargs)

    def checked_distribution(*args, **kwargs):
        assert not torch.is_grad_enabled()
        calls["distribution"] += 1
        return distribution(*args, **kwargs)

    monkeypatch.setattr(model, "encode", checked_encode)
    monkeypatch.setattr(model, "distribution", checked_distribution)
    proposals = model.proposals(domains, max_candidates=4)
    assert next(proposals) == expected and torch.is_grad_enabled()
    candidates = [expected, *proposals]
    assert len(candidates) == 4 and calls["encode"] == 1
    assert candidates == list(model.proposals(domains, max_candidates=4))
    assert torch.equal(before, torch.get_rng_state()) and torch.is_grad_enabled()
    assert all(parameter.grad is None for parameter in model.parameters())


def test_proposals_supply_32_distinct_constructions_when_available():
    model = _model(contextual=True, target_first=True, max_cores=3, max_open_colors=2).eval()
    model.beta = 0
    candidates = list(model.proposals([[0, 1]] * 5))
    assert len(candidates) == len({(candidate.core_keys, candidate.open_colors) for candidate in candidates}) == 32


def test_proposals_deduplicate_core_and_output_orders_without_losing_target_anchors(monkeypatch):
    torch.manual_seed(13)
    model = _proposal_model(contextual=True, target_first=True)
    model.beta = 0
    distribution, visited = model.distribution, []

    def record(encoded, sizes, keys, opened, output_phase, beta, targets):
        visited.append((frozenset(keys), frozenset(opened), output_phase, opened[0] if opened else None))
        return distribution(encoded, sizes, keys, opened, output_phase, beta, targets)

    monkeypatch.setattr(model, "distribution", record)
    candidates = list(model.proposals([[0, 1]] * 3, max_candidates=100, max_steps=1000))
    signatures = {(candidate.core_keys, candidate.open_colors) for candidate in candidates}
    # Two single-core scopes contribute three nonempty output sets each; their
    # connected union contributes six. Different traces must not consume slots.
    assert len(candidates) == len(signatures) == 12
    assert len(visited) == len(set(visited))
    a, b = model.color_idx["a"], model.color_idx["b"]
    for anchor in (a, b):
        assert (frozenset((0, 1)), frozenset((a, b)), True, anchor) in visited


def test_proposals_continue_past_skip_and_bound_distribution_evaluations(monkeypatch):
    model = _proposal_model()
    model.beta = 0
    distribution, calls = model.distribution, []

    def prefer_skip(*args, **kwargs):
        actions, _ = distribution(*args, **kwargs)
        calls.append(1)
        logits = torch.tensor([10. if kind == "skip" else 0. for kind, _ in actions])
        return actions, torch.distributions.Categorical(logits=logits)

    monkeypatch.setattr(model, "distribution", prefer_skip)
    assert model.propose([[0, 1]] * 3) is None
    candidates = list(model.proposals([[0, 1]] * 3, max_candidates=100, max_steps=1000))
    # Empty outputs are legal outside target-first construction.
    assert len(candidates) == 15 and any(not candidate.open_colors for candidate in candidates)
    calls.clear()
    assert list(model.proposals([[0, 1]] * 3, max_steps=1)) == [] and len(calls) == 1


@pytest.mark.parametrize("limits", [dict(max_candidates=0), dict(max_candidates=True),
                                    dict(max_steps=-1), dict(max_steps=1.5)])
def test_proposals_reject_invalid_limits(limits):
    with pytest.raises(ValueError, match="positive integers"):
        list(_proposal_model().proposals([[0, 1]] * 3, **limits))


def test_selected_topology_distinguishes_and_learns_the_odd_cycle_extension():
    torch.manual_seed(13)
    model = _model(contextual=True, exploration=0)
    state = _k5().create_mac_state()
    assert state.propagate()
    selected = ["01", "12", "23", "34"]
    keys = [model.key_idx[key] for key in selected]
    # This establishes the diagnostic's actual contraction consequences.
    for extension, contradiction in (("02", True), ("30", False)):
        candidate = SummaryClusterCandidate(extension, tuple(selected + [extension]), ())
        assert evaluate_candidate(state, candidate).contradiction == contradiction
    encoded = model.encode(state.domains)
    assert (encoded[1] - encoded[1][0]).abs().max() == 0
    _, contextual = model.selected_encoding(encoded, keys, [])
    assert (contextual[model.key_idx["02"]] - contextual[model.key_idx["30"]]).abs().max() > 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    for _ in range(60):
        optimizer.zero_grad()
        actions, distribution = model.distribution(model.encode(state.domains), [2] * 5, keys, [], False, 0)
        good, bad = [actions.index(("core", model.key_idx[key])) for key in ("02", "30")]
        loss = torch.nn.functional.softplus(distribution.logits[bad] - distribution.logits[good])
        loss.backward()
        optimizer.step()
    assert distribution.logits[good] - distribution.logits[bad] > 1
    for module in (model.selection_variable, model.selection_to_core, model.extension_core):
        assert module[0].weight.grad is not None and torch.isfinite(module[0].weight.grad).all()


def test_guidance_leaves_novel_construction_probability_and_no_repeated_skip():
    model = _model(contextual=True, max_cores=32)
    domains = [[0, 1]] * 5
    targets = ((frozenset((0, 1, 2)), frozenset((0,))),)
    encoded = model.encode(domains)
    actions, distribution = model.distribution(encoded, [2] * 5, [0], [], False, 100, targets)
    assert ("skip", -1) not in actions
    novel = [idx for idx, (kind, key) in enumerate(actions) if kind == "core" and key not in targets[0][0]]
    assert novel and distribution.probs[novel].sum() > .05
    # An off-proposal prefix is completed in the unrestricted core/open space.
    candidate, _, _, trace = model.rollout(domains, prefix=[("core", 7)], beta=0, greedy=True)
    assert trace[0] == ("core", 7)
    assert candidate is not None and model.keys[7] in candidate.core_keys
    assert ("finish", -1) in model.actions([0], [], True, [2] * 5)  # Scalar summaries remain legal.


def test_teacher_trace_scores_without_sampling_and_prefix_scores_only_suffix():
    model = _model(contextual=True)
    domains = [[0, 1]] * 5
    candidate = SummaryClusterCandidate("teacher", ("01", "02", "12"), ("0",))
    trace = model.candidate_trace(candidate, domains)
    rng = torch.get_rng_state().clone()
    constructed, full_log_prob, _, actual = model.rollout(domains, trace=trace, beta=0)
    assert torch.equal(rng, torch.get_rng_state()) and actual == trace
    assert set(constructed.core_keys) == set(candidate.core_keys) and constructed.open_colors == ("0",)
    full_log_prob.backward()
    assert model.head[-1].weight.grad.abs().sum() > 0
    completed, suffix_log_prob, _, _ = model.rollout(domains, prefix=trace[:-1], beta=0, greedy=True)
    assert completed is not None and suffix_log_prob > full_log_prob
    _, zero, _, _ = model.rollout(domains, prefix=trace, beta=0)
    assert zero == 0
    with pytest.raises(ValueError, match="ended before"):
        model.rollout(domains, trace=trace[:-1], beta=0)
    with pytest.raises(ValueError, match="after completion"):
        model.rollout(domains, trace=trace + [("skip", -1)], beta=0)
    with pytest.raises(ValueError, match="Illegal"):
        model.rollout(domains, trace=[("core", 0), ("core", 0)], beta=0)


def test_teacher_random_orders_cover_partial_sets_and_preserve_the_construction():
    model = _model(contextual=True)
    domains = [[0, 1]] * 5
    candidate = SummaryClusterCandidate("teacher", ("01", "02", "12"), ("0", "1", "2"))
    rng, replay = Random(13), Random(13)
    first, outputs = set(), set()
    for _ in range(12):
        trace = model.candidate_trace(candidate, domains, rng=rng)
        assert trace == model.candidate_trace(candidate, domains, rng=replay)
        first.add(trace[0])
        outputs.add(tuple(idx for kind, idx in trace if kind == "open"))
        actual, _, _, decisions = model.rollout(domains, trace=trace, beta=0)
        assert actual.core_keys == candidate.core_keys and actual.open_colors == candidate.open_colors
        assert decisions == trace
    assert len(first) == 3 and len(outputs) > 1


def test_active_frontier_does_not_grow_through_a_fixed_variable():
    network = ConstraintNetwork({key: NumpyCore(values=np.ones((2, 2)), colors=list(key))
                                 for key in ("ab", "bc", "cd")}, maintain_indices=True)
    model = Policy(network_data(network), width=8, rounds=1, contextual=True)
    assert ("core", model.key_idx["bc"]) in model.actions([model.key_idx["ab"]], [], False)
    sizes = [2, 1, 2, 2]
    assert ("core", model.key_idx["bc"]) not in model.actions([model.key_idx["ab"]], [], False, sizes)
    assert ("open", model.color_idx["b"]) not in model.actions([model.key_idx["ab"]], [], True, sizes)


def test_batched_action_features_match_scalar_reference():
    model = _model(contextual=True)
    keys = [model.key_idx[key] for key in ("01", "12", "23", "34")]
    scope, sizes = set(range(5)), [2, 1, 2, 2, 2]
    captured = []
    hook = model.head.register_forward_pre_hook(lambda _, inputs: captured.append(inputs[0].detach()))
    actions, _ = model.distribution(model.encode([[0, 1]] * 5), sizes, keys, [], False, 0)
    hook.remove()
    degree = [sum(color in model.scopes[key] for key in keys) for color in range(5)]
    neighbor = []
    for color in range(5):
        neighbors = [other for key in keys if color in model.scopes[key]
                     for other in model.scopes[key] if other != color]
        neighbor.append(sum(degree[other] for other in neighbors) / max(1, len(neighbors)))
    expected = []
    for kind, idx in actions:
        colors = model.scopes[idx] if kind == "core" else [idx] if kind == "open" else []
        pressure = sum(1 - sizes[c] / 2 for c in colors if kind == "open" or sizes[c] > 1) / max(1, len(colors))
        overlap = min(sum(c in scope and sizes[c] > 1 for c in colors), 2) / 2 if kind == "core" else 0
        topology = [sum(values) / max(1, len(colors)) for values in (
            [log1p(degree[c]) for c in colors], [log1p(neighbor[c]) for c in colors],
            [float(degree[c] == 1) for c in colors], [float(degree[c] > 0) for c in colors])]
        expected.append([0, len(keys) / model.config["max_cores"], 0, 0, len(scope) / 52,
                         pressure, overlap, *[float(kind == label) for label in ("core", "open", "finish", "skip")],
                         *topology])
    torch.testing.assert_close(captured[0][:, 4 * model.config["width"]:], torch.tensor(expected))


def test_closed_state_one_active_core_is_irrelevant_and_optional_mask_excludes_it():
    network = ConstraintNetwork({
        "fixed": NumpyCore(values=np.array([[1, 0], [0, 0]]), colors=["x", "y"]),
        "one": NumpyCore(values=np.array([[1, 1], [0, 1]]), colors=["x", "a"]),
        "two": NumpyCore(values=np.eye(2, dtype=bool), colors=["a", "b"])}, maintain_indices=True)
    state = network.create_mac_state(dict(x=(0,), y=(0,), a=(0, 1), b=(0, 1)))
    assert state.propagate()
    trial = evaluate_candidate(state, SummaryClusterCandidate("one", ("one",), ("a",)))
    assert not trial.contradiction and trial.removed_domain_values == 0
    model = Policy(network_data(network), width=8, rounds=1, contextual=True, min_active=2)
    sizes = [len(state.domains[color]) for color in model.network["colors"]]
    actions = model.actions([], [], False, sizes)
    assert ("core", model.key_idx["two"]) in actions
    assert ("core", model.key_idx["one"]) not in actions
    assert ("core", model.key_idx["fixed"]) not in actions
    legacy_context = Policy(model.network, width=8, rounds=1, contextual=True)
    assert ("core", model.key_idx["one"]) in legacy_context.actions([], [], False, sizes)


@pytest.mark.parametrize("hypergraph", [False, True])
def test_gac_closed_factor_forest_cannot_prune_and_only_optional_mask_forbids_finishing(hypergraph):
    if hypergraph:
        parity = np.fromfunction(lambda a, b, c: (a + b + c) % 2 == 0, (2, 2, 2))
        cores = {"abc": NumpyCore(values=parity, colors=list("abc")),
                 "cde": NumpyCore(values=parity, colors=list("cde"))}
    else:
        cores = {"ab": NumpyCore(values=np.eye(2, dtype=bool), colors=list("ab")),
                 "bc": NumpyCore(values=~np.eye(2, dtype=bool), colors=list("bc")),
                 "bd": NumpyCore(values=~np.eye(2, dtype=bool), colors=list("bd"))}
    network = ConstraintNetwork(cores, maintain_indices=True)
    state = network.create_mac_state()
    assert state.propagate()
    candidate = SummaryClusterCandidate("forest", tuple(cores), tuple(network.dimension_dict))
    result = evaluate_candidate(state, candidate)
    assert not result.contradiction and result.removed_domain_values == 0
    model = Policy(network_data(network), width=8, rounds=1, contextual=True, cyclic_only=True,
                   max_cores=len(cores), max_open_colors=5)
    selected = list(range(len(cores)))
    assert model._active_cycle_rank(selected, [2] * len(model.network["colors"])) == 0
    assert model.actions(selected, [], False, [2] * len(model.network["colors"])) == [("skip", -1)]
    # Finishing the output phase, including scalar output, remains legal.
    assert ("finish", -1) in model.actions(selected, [], True, [2] * len(model.network["colors"]))
    with pytest.raises(ValueError, match="cannot finish"):
        model.candidate_trace(candidate, state.domains)
    model.config["cyclic_only"] = False
    assert ("finish", -1) in model.actions(selected, [], False)


def test_cycle_mask_uses_active_scopes_and_accepts_both_odd_and_even_cycles():
    for length in (3, 4):
        cores = {f"e{i}": NumpyCore(values=~np.eye(2, dtype=bool),
                  colors=[str(i), str((i + 1) % length)]) for i in range(length)}
        model = Policy(network_data(ConstraintNetwork(cores, maintain_indices=True)), width=8, rounds=1,
                       contextual=True, cyclic_only=True, max_cores=length)
        keys = list(range(length))
        assert model._active_cycle_rank(keys, [2] * length) == 1
        assert ("finish", -1) in model.actions(keys, [], False, [2] * length)
        assert model._active_cycle_rank(keys, [1] + [2] * (length - 1)) == 0
        assert ("finish", -1) not in model.actions(keys, [], False, [1] + [2] * (length - 1))


@pytest.mark.parametrize("kind", ["core", "open"])
def test_hierarchical_greedy_compares_total_kind_mass_before_selecting_member(kind):
    model = _model(contextual=True)
    actions = [("finish", -1), (kind, 0), (kind, 1), (kind, 2)]
    distribution = torch.distributions.Categorical(probs=torch.tensor([.4, .25, .2, .15]))
    assert int(model._greedy_choice(actions, distribution)) == 0
    model.config["hierarchical_greedy"] = True
    assert int(model._greedy_choice(actions, distribution)) == 1
    # It still stops when stopping is the most probable action kind.
    assert int(model._greedy_choice(actions, torch.distributions.Categorical(
        probs=torch.tensor([.6, .15, .15, .1])))) == 0


def test_hierarchical_decode_option_preserves_sampling_and_teacher_probabilities(tmp_path):
    torch.manual_seed(13)
    model = _model(contextual=True)
    assert "hierarchical_greedy" not in model.config
    domains = [[0, 1]] * 5
    candidate = SummaryClusterCandidate("teacher", ("01", "02", "12"), ("0",))
    trace = model.candidate_trace(candidate, domains)
    plain = model.rollout(domains, trace=trace, beta=0)
    torch.manual_seed(42)
    sampled = model.rollout(domains, beta=0)
    model.config["hierarchical_greedy"] = True
    hierarchical = model.rollout(domains, trace=trace, beta=0)
    torch.manual_seed(42)
    repeated = model.rollout(domains, beta=0)
    assert sampled[3] == repeated[3]
    torch.testing.assert_close(plain[1], hierarchical[1], atol=0, rtol=0)
    torch.testing.assert_close(sampled[1], repeated[1], atol=0, rtol=0)
    path = tmp_path / "hierarchical.pt"
    torch.save(dict(version=1, kind=model.kind, network=model.network, config=model.config,
                    reward_colors=None, weights=model.state_dict(), beta=0), path)
    assert runtime.load(path).config["hierarchical_greedy"] is True


def test_target_first_selects_active_output_then_incident_core_and_keeps_target():
    network = ConstraintNetwork({key: NumpyCore(values=np.ones((2, 2)), colors=list(key))
                                 for key in ("ab", "bc", "cd", "da")}, maintain_indices=True)
    model = Policy(network_data(network), width=8, rounds=1, contextual=True, target_first=True,
                   min_active=2, cyclic_only=True)
    model.reward_colors = ["a"]  # Auxiliary targets remain available.
    domains = {color: (0, 1) for color in "abcd"}
    sizes = [2] * 4
    start = model.actions([], [], False, sizes)
    assert start == [("skip", -1)] + [("open", idx) for idx in range(4)]
    target = model.color_idx["b"]
    after_target = model.actions([], [target], False, sizes)
    assert {idx for kind, idx in after_target if kind == "core"} == {model.key_idx["ab"], model.key_idx["bc"]}
    assert not any(kind in ("finish", "open") for kind, _ in after_target)
    # Construction can grow away from the target after the incident seed.
    assert ("core", model.key_idx["cd"]) in model.actions([model.key_idx["bc"]], [target], False, sizes)
    candidate = SummaryClusterCandidate("cycle", tuple(model.keys), ("a", "b"))
    trace = model.candidate_trace(candidate, domains, target_color="b", rng=Random(13))
    assert trace[0] == ("open", target) and trace[1][0] == "core"
    assert target in model.scope_sets[trace[1][1]]
    actual, _, _, decisions = model.rollout(domains, trace=trace, beta=0)
    assert actual.open_colors == candidate.open_colors and actual.core_keys == candidate.core_keys
    assert sum(action == ("open", target) for action in decisions) == 1
    assert all(model.color_idx[color] in set().union(*(model.scope_sets[model.key_idx[key]] for key in actual.core_keys))
               for color in actual.open_colors)
    with pytest.raises(ValueError, match="Illegal"):
        model.rollout(domains, trace=[("open", target), ("core", model.key_idx["cd"])], beta=0)


def test_target_first_teacher_prefers_reward_target_but_can_use_auxiliary_and_rejects_scalar():
    model = _model(contextual=True, target_first=True, cyclic_only=True)
    model.reward_colors = ["2"]
    domains = [[0, 1]] * 5
    candidate = SummaryClusterCandidate("teacher", ("01", "02", "12"), ("0", "1", "2"))
    assert model.candidate_trace(candidate, domains)[0] == ("open", model.color_idx["2"])
    auxiliary = SummaryClusterCandidate("teacher", candidate.core_keys, ("0", "1"))
    trace = model.candidate_trace(auxiliary, domains, rng=Random(13))
    assert trace[0][1] in {model.color_idx["0"], model.color_idx["1"]}
    assert model.rollout(domains, trace=trace, beta=0)[0].open_colors == auxiliary.open_colors
    with pytest.raises(ValueError, match="requires an active"):
        model.candidate_trace(SummaryClusterCandidate("scalar", candidate.core_keys, ()), domains)
    with pytest.raises(ValueError, match="Requested target"):
        model.candidate_trace(candidate, domains, target_color="4")


def test_random_target_teacher_traces_cover_primary_and_auxiliary_goals():
    model = _model(contextual=True, target_first=True, cyclic_only=True)
    model.reward_colors = ["2"]
    domains = [[0, 1]] * 5
    candidate = SummaryClusterCandidate("mixed", ("01", "02", "12"), ("0", "2"))
    rng, replay = Random(13), Random(13)
    targets = set()
    for _ in range(16):
        trace = model.candidate_trace(candidate, domains, rng=rng)
        assert trace == model.candidate_trace(candidate, domains, rng=replay)
        kind, target = trace[0]
        targets.add(target)
        assert kind == "open" and trace[1][0] == "core"
        assert target in model.scope_sets[trace[1][1]]
        assert model.rollout(domains, trace=trace, beta=0)[0].open_colors == candidate.open_colors
    assert targets == {model.color_idx["0"], model.color_idx["2"]}
    assert model.candidate_trace(candidate, domains, rng=Random(13), target_color="2")[0] == ("open", model.color_idx["2"])


def test_target_first_masks_fixed_oversized_and_dead_end_outputs_and_roundtrips(tmp_path):
    cores = {"ab": NumpyCore(values=np.ones((2, 2)), colors=list("ab")),
             "cd": NumpyCore(values=np.ones((5, 2)), colors=list("cd"))}
    model = Policy(network_data(ConstraintNetwork(cores, maintain_indices=True)), width=8, rounds=1,
                   contextual=True, target_first=True, min_active=2, max_size=4, compact_domains=True)
    sizes = [2, 1, 2, 2]
    # a has no eligible incident factor once b is fixed; c's expanded output is
    # too large. Only d remains, and its first core must contain it.
    assert model.actions([], [], False, sizes) == [("skip", -1), ("open", model.color_idx["d"])]
    path = tmp_path / "target.pt"
    torch.save(dict(version=1, kind=model.kind, network=model.network, config=model.config,
                    reward_colors=None, weights=model.state_dict(), beta=0), path)
    restored = runtime.load(path)
    assert restored.config == model.config and restored.config["target_first"]
    assert restored.actions([], [], False, sizes) == model.actions([], [], False, sizes)
    assert "target_first" not in _model().config


def test_contextual_checkpoint_roundtrip(tmp_path):
    model = _model(contextual=True, max_cores=32, compact_domains=True, max_total_work=12345).eval()
    path = tmp_path / "contextual.pt"
    torch.save(dict(version=1, kind=model.kind, network=model.network, config=model.config,
                    reward_colors=None, weights=model.state_dict(), beta=0), path)
    restored = runtime.load(path)
    assert restored.config == model.config and restored.contextual
    expected = model.rollout([[0, 1]] * 5, beta=0, greedy=True)
    actual = restored.rollout([[0, 1]] * 5, beta=0, greedy=True)
    assert expected[3] == actual[3]
    torch.testing.assert_close(expected[1], actual[1], atol=0, rtol=0)


def _value_problem(*, permuted=False, reordered=False):
    rng = np.random.default_rng(71)
    dimensions = dict(a=4, b=3, c=4, d=2)
    domains = dict(a=(0, 1, 3), b=(1, 2), c=(0, 2), d=(1,))
    cores = [(key, tuple(key), rng.random(tuple(dimensions[color] for color in key)) < .7)
             for key in ("ab", "bc", "ac", "cd", "abd")]
    labels = {color: rng.permutation(dim) if permuted else np.arange(dim)
              for color, dim in dimensions.items()}
    transformed = {}
    for key, colors, table in reversed(cores) if reordered else cores:
        for axis, color in enumerate(colors):
            table = np.take(table, np.argsort(labels[color]), axis=axis)
        order = tuple(reversed(range(len(colors)))) if reordered else tuple(range(len(colors)))
        transformed[key] = NumpyCore(values=table.transpose(order), colors=[colors[axis] for axis in order])
    return network_data(ConstraintNetwork(transformed, maintain_indices=True)), {
        color: tuple(int(labels[color][value]) for value in values) for color, values in domains.items()}


def test_value_encoding_remains_an_explicit_opt_in_with_unchanged_legacy_parameters():
    torch.manual_seed(17)
    default = _model(contextual=True)
    torch.manual_seed(17)
    disabled = _model(contextual=True, value_encoding=False)
    assert default.config == disabled.config and "value_encoding" not in default.config
    assert default.state_dict().keys() == disabled.state_dict().keys()
    for key in default.state_dict():
        torch.testing.assert_close(default.state_dict()[key], disabled.state_dict()[key], atol=0, rtol=0)


@pytest.mark.parametrize("permuted,reordered", [(True, False), (False, True), (True, True)])
def test_value_encoding_is_invariant_to_independent_labels_core_order_and_factor_axes(permuted, reordered):
    raw, domains = _value_problem()
    changed, changed_domains = _value_problem(permuted=permuted, reordered=reordered)
    options = dict(width=8, rounds=2, contextual=True, target_first=True, value_encoding=True,
                   exploration=0., min_active=2, max_cores=5, max_open_colors=3, heuristic_candidates=0)
    torch.manual_seed(29)
    model, renamed = Policy(raw, **options).eval(), Policy(changed, **options).eval()
    renamed.load_state_dict(model.state_dict())
    encoded, renamed_encoded = model.encode(domains), renamed.encode(changed_domains)
    for color in raw["colors"]:
        torch.testing.assert_close(encoded[0][model.color_idx[color]],
            renamed_encoded[0][renamed.color_idx[color]], atol=1e-6, rtol=0)
    for key in model.keys:
        torch.testing.assert_close(encoded[1][model.key_idx[key]],
            renamed_encoded[1][renamed.key_idx[key]], atol=1e-6, rtol=0)

    def probabilities(policy, encoded, ds, keys, opened, phase):
        actions, distribution = policy.distribution(encoded,
            [len(ds[color]) for color in policy.network["colors"]],
            [policy.key_idx[key] for key in keys], [policy.color_idx[color] for color in opened], phase, 0.)
        return {(kind, policy.keys[idx] if kind == "core" else policy.network["colors"][idx]
                 if kind == "open" else idx): probability
                for (kind, idx), probability in zip(actions, distribution.probs)}

    for keys, opened, phase in (((), (), False), ((), ("a",), False),
                                (("ab", "bc"), ("a",), False), (("ab", "bc", "ac"), ("a",), True)):
        left = probabilities(model, encoded, domains, keys, opened, phase)
        right = probabilities(renamed, renamed_encoded, changed_domains, keys, opened, phase)
        assert left.keys() == right.keys()
        for action in left:
            torch.testing.assert_close(left[action], right[action], atol=1e-6, rtol=0)


def test_value_encoder_weights_transfer_from_four_to_nine_values_with_equivalent_active_constraints(tmp_path):
    closing = np.ones((4, 4), dtype=bool)
    closing[1, 1] = False
    original = {"ab": np.eye(4, dtype=bool), "bc": np.eye(4, dtype=bool), "ac": closing}
    injection = dict(a=np.array([7, 1, 5, 3]), b=np.array([4, 8, 0, 6]), c=np.array([2, 5, 7, 1]))
    small = ConstraintNetwork({key: NumpyCore(values=table, colors=list(key)) for key, table in original.items()},
                              maintain_indices=True)
    cores = {}
    for key, table in original.items():
        values = np.ones((9, 9), dtype=bool)
        values[np.ix_(*(injection[color] for color in key))] = table
        cores[key] = NumpyCore(values=values, colors=list(key))
    large = ConstraintNetwork(cores, maintain_indices=True)
    options = dict(width=8, rounds=2, contextual=True, target_first=True, value_encoding=True,
                   compact_domains=True, max_cores=3, max_open_colors=1, heuristic_candidates=0)
    model, transferred = Policy(network_data(small), **options), Policy(network_data(large), **options)
    assert {key: tuple(value.shape) for key, value in model.named_parameters()} == {
        key: tuple(value.shape) for key, value in transferred.named_parameters()}
    assert not any(name.startswith(("variable_encoder.", "core_encoder.", "to_core.", "to_variable."))
                   for name, _ in model.named_parameters())
    transferred.load_state_dict(model.state_dict())
    states = [small.create_mac_state({color: (0, 1) for color in "abc"}),
              large.create_mac_state({color: tuple(map(int, injection[color][:2])) for color in "abc"})]
    candidate = SummaryClusterCandidate("triangle", ("ab", "bc", "ac"), ("a",))
    for policy, state in zip((model, transferred), states):
        assert state.propagate()
        encoded = policy.encode(state.domains)
        assert all(torch.isfinite(values).all() for values in encoded)
        result = evaluate_candidate(state, candidate, compact_domains=True)
        assert result.removed_domain_values == 3
        expected = 0 if policy is model else int(injection["a"][0])
        assert result.domains["a"] == (expected,)
    path = tmp_path / "values.pt"
    torch.save(dict(version=1, kind=transferred.kind, network=transferred.network, config=transferred.config,
                    reward_colors=None, weights=transferred.state_dict(), beta=0), path)
    restored = runtime.load(path)
    assert restored.config["value_encoding"]
    for expected, actual in zip(transferred.encode(states[1].domains), restored.encode(states[1].domains)):
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)


def test_shared_value_encoder_learns_a_cluster_with_real_generic_pruning():
    from experiments.cluster_selection.reinforcement import learn
    closing = np.ones((4, 4), dtype=bool)
    closing[1, 1] = False
    network = ConstraintNetwork({key: NumpyCore(values=values, colors=list(key)) for key, values in (
        ("ab", np.eye(4, dtype=bool)), ("bc", np.eye(4, dtype=bool)), ("ac", closing),
        ("ax", np.ones((4, 4), dtype=bool)), ("xy", np.ones((4, 4), dtype=bool)),
        ("ay", np.ones((4, 4), dtype=bool)))}, maintain_indices=True)
    state = network.create_mac_state({color: (0, 1) for color in network.dimension_dict})
    assert state.propagate()
    before = state.snapshot_domains(), tuple(state._cores), tuple(state._trail), state.statistics
    torch.manual_seed(19)
    model = Policy(network_data(network), width=8, rounds=2, contextual=True, target_first=True,
                   value_encoding=True, exploration=0., compact_domains=True, min_active=2,
                   cyclic_only=True, max_cores=3, max_open_colors=1, heuristic_candidates=0)
    model.reward_colors = list(state.domains)
    teacher = SummaryClusterCandidate("useful", ("ab", "bc", "ac"), ("a",))
    decoy = SummaryClusterCandidate("decoy", ("ax", "xy", "ay"), ("a",))
    metrics = learn.measure(model, state, teacher, known_satisfiable=True)
    assert metrics["useful"] and metrics["gain"] == 3
    assert evaluate_candidate(state, decoy, compact_domains=True).removed_domain_values == 0
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    losses = []
    for step in range(40):
        optimizer.zero_grad()
        loss = learn.teacher_loss(model, state.domains, [(teacher, metrics)], Random(step))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
        if step == 0:
            for module in (model.value_encoder, model.value_core_encoder, model.value_to_tuple, model.tuple_to_value):
                assert module[0].weight.grad is not None and module[0].weight.grad.abs().sum() > 0
        losses.append(float(loss.detach()))
        optimizer.step()
    assert losses[-1] < losses[0] / 2
    model.eval()
    candidate = model.propose(state.domains)
    assert candidate is not None
    assert evaluate_candidate(state, candidate, compact_domains=True).removed_domain_values == 3
    assert (state.snapshot_domains(), tuple(state._cores), tuple(state._trail), state.statistics) == before
    state.push_level()
    assert runtime.apply_candidate(state, candidate, compact_domains=True)
    assert state.domains["a"] == (0,) and not state._temporary_core_keys
    state.pop_level()
    assert (state.snapshot_domains(), tuple(state._cores), tuple(state._trail)) == before[:3]
    assert state.decision_level == 0 and not state._added_core_keys


def _relational_problem(*, permuted=False, reordered=False):
    identity = np.eye(3, dtype=bool)
    relations = [("ab", ("a", "b"), identity), ("bc", ("b", "c"), identity),
                 ("identity", ("a", "c"), identity), ("swap", ("a", "c"), identity[[1, 0, 2]])]
    labels = dict(a=np.array([2, 0, 1]), b=np.array([1, 2, 0]), c=np.array([2, 1, 0]))
    cores = {}
    for key, colors, table in reversed(relations) if reordered else relations:
        if permuted:
            for axis, color in enumerate(colors):
                table = np.take(table, np.argsort(labels[color]), axis=axis)
        if reordered:
            table, colors = table.T, colors[::-1]
        cores[key] = NumpyCore(values=table, colors=list(colors))
    return ConstraintNetwork(cores, maintain_indices=True)


def test_relational_context_distinguishes_useful_cycle_composition_and_has_finite_gradients():
    network = _relational_problem()
    state = network.create_mac_state()
    assert state.propagate()
    candidates = [SummaryClusterCandidate(key, ("ab", "bc", key), ("a",)) for key in ("identity", "swap")]
    assert [evaluate_candidate(state, candidate, compact_domains=True).removed_domain_values
            for candidate in candidates] == [0, 6]
    options = dict(width=16, rounds=2, contextual=True, value_encoding=True, target_first=True,
                   exploration=0., max_cores=3, max_open_colors=1, cyclic_only=True, heuristic_candidates=0)
    torch.manual_seed(13)
    old = Policy(network_data(network), **options).eval()
    keys, opened = [old.key_idx[key] for key in ("ab", "bc")], [old.color_idx["a"]]
    old_cores = old.selected_encoding(old.encode(state.domains), keys, opened)[1]
    torch.testing.assert_close(old_cores[old.key_idx["identity"]], old_cores[old.key_idx["swap"]], atol=0, rtol=0)
    torch.manual_seed(13)
    model = Policy(network_data(network), relational_context=True, relational_rounds=6, **options).eval()
    encoded = model.encode(state.domains)
    selected = model.selected_encoding(encoded, keys, opened)[1]
    assert (selected[model.key_idx["identity"]] - selected[model.key_idx["swap"]]).abs().max() > 1e-5
    actions, distribution = model.distribution(encoded, [3] * 3, keys, opened, False, 0.)
    good, bad = [actions.index(("core", model.key_idx[key])) for key in ("swap", "identity")]
    assert abs(float(distribution.logits[good] - distribution.logits[bad])) > 1e-8
    (-distribution.logits[good]).backward()
    parameters = {name: parameter for name, parameter in model.named_parameters() if name.startswith("relational_")}
    assert parameters
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters.values())
    assert any(parameter.grad.abs().sum() > 0 for parameter in parameters.values())


@pytest.mark.parametrize("permuted,reordered", [(True, False), (False, True), (True, True)])
def test_relational_context_is_invariant_to_value_labels_core_order_and_axes(permuted, reordered):
    networks = [_relational_problem(), _relational_problem(permuted=permuted, reordered=reordered)]
    options = dict(width=16, rounds=2, contextual=True, value_encoding=True, target_first=True,
                   relational_context=True, relational_rounds=6, exploration=0., max_cores=3,
                   max_open_colors=1, cyclic_only=True, heuristic_candidates=0)
    torch.manual_seed(17)
    models = [Policy(network_data(network), **options).eval() for network in networks]
    models[1].load_state_dict(models[0].state_dict())
    results = []
    for model, network in zip(models, networks):
        state = network.create_mac_state()
        assert state.propagate()
        encoded = model.encode(state.domains)
        keys, opened = [model.key_idx[key] for key in ("ab", "bc")], [model.color_idx["a"]]
        variables, cores = model.selected_encoding(encoded, keys, opened)
        actions, distribution = model.distribution(encoded, [3] * 3, keys, opened, False, 0.)
        results.append((model, variables, cores, {(kind, model.keys[idx]): probability
                        for (kind, idx), probability in zip(actions, distribution.probs)}))
    left, right = results
    for color in "abc":
        torch.testing.assert_close(left[1][left[0].color_idx[color]], right[1][right[0].color_idx[color]], atol=1e-6, rtol=0)
    for key in left[0].keys:
        torch.testing.assert_close(left[2][left[0].key_idx[key]], right[2][right[0].key_idx[key]], atol=1e-6, rtol=0)
    assert left[3].keys() == right[3].keys()
    for action in left[3]:
        torch.testing.assert_close(left[3][action], right[3][action], atol=1e-6, rtol=0)


def test_relational_context_opt_in_preserves_defaults_and_roundtrips_latest_configuration(tmp_path):
    torch.manual_seed(19)
    default = _model(contextual=True)
    torch.manual_seed(19)
    disabled = _model(contextual=True, relational_context=False)
    assert default.config == disabled.config and "relational_context" not in default.config
    assert "relational_rounds" not in default.config and len(default.encode([[0, 1]] * 5)) == 2
    for key in default.state_dict():
        torch.testing.assert_close(default.state_dict()[key], disabled.state_dict()[key], atol=0, rtol=0)
    network = _relational_problem()
    options = dict(width=16, rounds=2, contextual=True, value_encoding=True, target_first=True,
                   relational_context=True, relational_rounds=6, exploration=0., max_cores=3,
                   max_open_colors=1, cyclic_only=True, heuristic_candidates=0)
    for required in ("contextual", "value_encoding", "target_first"):
        with pytest.raises(ValueError, match="requires"):
            Policy(network_data(network), **{**options, required: False})
    model = Policy(network_data(network), **options).eval()
    path = tmp_path / "relational.pt"
    torch.save(dict(version=1, kind=model.kind, network=model.network, config=model.config,
                    reward_colors=None, weights=model.state_dict(), beta=0), path)
    restored = runtime.load(path)
    assert restored.config == model.config and restored.config["relational_rounds"] == 6
    domains = network.create_mac_state().domains
    prefix = [("open", model.color_idx["a"]), *[("core", model.key_idx[key]) for key in ("ab", "bc")]]
    expected = model.rollout(domains, prefix=prefix, beta=0., greedy=True)
    actual = restored.rollout(domains, prefix=prefix, beta=0., greedy=True)
    assert expected[3] == actual[3] and expected[0] == actual[0]
    torch.testing.assert_close(expected[1], actual[1], atol=0, rtol=0)


def test_q_learning_opt_in_preserves_actor_defaults_and_weight_shapes():
    torch.manual_seed(31)
    actor = _proposal_model(contextual=True)
    torch.manual_seed(31)
    disabled = _proposal_model(contextual=True, q_learning=False)
    torch.manual_seed(31)
    q_model = _proposal_model(contextual=True, q_learning=True)
    assert actor.config == disabled.config and "q_learning" not in actor.config
    assert q_model.config["q_learning"] is True
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, disabled.state_dict()[key], atol=0, rtol=0)
        torch.testing.assert_close(value, q_model.state_dict()[key], atol=0, rtol=0)
    domains = dict(a=(0, 1), b=(0, 1), c=(0, 1))
    left, right = [model.rollout(domains, beta=0., greedy=True) for model in (actor, disabled)]
    assert left[0] == right[0] and left[3] == right[3]
    torch.testing.assert_close(left[1], right[1], atol=0, rtol=0)


@pytest.mark.parametrize("options", [dict(q_learning=True), dict(contextual=True, q_learning=1),
                                     dict(contextual=True, q_learning="true")])
def test_q_learning_requires_contextual_encoding_and_boolean_configuration(options):
    with pytest.raises(ValueError, match="Q-learning requires"):
        _proposal_model(**options)


def test_q_values_require_contextual_encoding():
    model = _proposal_model()
    with pytest.raises(ValueError, match="Q values require contextual"):
        model.q_values(model.encode([[0, 1]] * 3), [2] * 3, [], [], False)


def test_q_values_retain_absolute_offset_gradients_and_skip_actor_biases_and_guidance(monkeypatch):
    model = _proposal_model(contextual=True, q_learning=True, exploration=.9).train()
    model.beta = 100.
    domains = dict(a=(0, 1), b=(0, 1), c=(0, 1))
    encoded, sizes = model.encode(domains), [2] * 3
    monkeypatch.setattr(model, "_prior", lambda *a, **k: pytest.fail("Q values must not use an actor prior"))
    with torch.no_grad():
        model.head[-1].weight.zero_()
        model.head[-1].bias.fill_(2.5)
    for keys in ([], [model.key_idx["ab"]]):
        actions, values = model.q_values(encoded, sizes, keys, [], False)
        assert actions == model.actions(keys, [], False, sizes)
        torch.testing.assert_close(values, torch.full_like(values, 2.5), atol=0, rtol=0)
        actual, distribution = model.distribution(encoded, sizes, keys, [], False, model.beta,
                                                   targets=(((0,), (0,)),))
        assert actual == actions
        torch.testing.assert_close(distribution.probs, torch.full_like(values, 1 / len(values)))
    values.sum().backward()
    torch.testing.assert_close(model.head[-1].bias.grad, torch.tensor([float(len(values))]))
    with torch.no_grad():
        model.head[-1].bias.add_(7.)
    _, shifted = model.q_values(encoded, sizes, keys, [], False)
    torch.testing.assert_close(shifted, values.detach() + 7., atol=0, rtol=0)


def test_q_inference_uses_individual_action_values_and_roundtrips_without_structural_targets(tmp_path,
                                                                                          monkeypatch):
    model = _proposal_model(contextual=True, q_learning=True, hierarchical_greedy=True)
    model.beta = 100.
    actions = [("core", 0), ("core", 1), ("finish", -1)]
    distribution = torch.distributions.Categorical(logits=torch.tensor([1., 1., 1.1]))
    assert int(model._greedy_choice(actions, distribution)) == 2
    monkeypatch.setattr(model, "_targets", lambda *a, **k: pytest.fail("Q inference must construct directly"))
    domains = dict(a=(0, 1), b=(0, 1), c=(0, 1))
    candidates = list(model.proposals(domains, max_candidates=4))
    assert len(candidates) == 4
    expected = model.propose(domains)
    path = tmp_path / "q.pt"
    torch.save(dict(version=1, kind=model.kind, network=model.network, config=model.config,
                    reward_colors=None, weights=model.state_dict(), beta=model.beta), path)
    restored = runtime.load(path)
    assert restored.config == model.config and restored.config["q_learning"] is True
    assert restored.propose(domains) == expected
    assert list(restored.proposals(domains, max_candidates=4)) == candidates
    actual_actions, values = model.q_values(model.encode(domains), [2] * 3, [], [], False)
    restored_actions, restored_values = restored.q_values(restored.encode(domains), [2] * 3, [], [], False)
    assert actual_actions == restored_actions
    torch.testing.assert_close(values, restored_values, atol=0, rtol=0)


def test_q_proposal_alternatives_rank_raw_values_when_softmax_probabilities_underflow(monkeypatch):
    model = _proposal_model(contextual=True, q_learning=True)
    model.config["max_cores"] = 1

    def scores(encoded, sizes, keys, opened, output_phase):
        actions = model.actions(keys, opened, output_phase, sizes)
        values = torch.zeros(len(actions))
        if not keys and not opened:
            values = torch.tensor([1000. if kind == "skip" else
                                   900. if idx == model.key_idx["bc"] else 800.
                                   for kind, idx in actions])
        return actions, values

    monkeypatch.setattr(model, "q_values", scores)
    candidate = next(model.proposals(dict(a=(0, 1), b=(0, 1), c=(0, 1)), max_candidates=1))
    assert candidate.core_keys == ("bc",)


_DEVICE_ENCODINGS = [("legacy", False), ("flat", False), ("flat", True),
                     ("value", False), ("value", True), ("relational", False), ("relational", True)]


def _device_policy(encoding, q_learning):
    return Policy(network_data(_relational_problem()), width=8, rounds=1, context_rounds=1,
        contextual=encoding != "legacy", target_first=encoding != "legacy", q_learning=q_learning,
        value_encoding=encoding in {"value", "relational"}, relational_context=encoding == "relational",
        relational_rounds=2, max_cores=3, max_open_colors=1, heuristic_candidates=0, cyclic_only=True)


def _device_scores(model, domains):
    encoded = model.encode(domains)
    sizes = [len(domains[color]) for color in model.network["colors"]]
    keys, opened = [model.key_idx[key] for key in ("ab", "bc")], [model.color_idx["a"]]
    if model.contextual:
        return model.q_values(encoded, sizes, keys, opened, False)
    actions, distribution = model.distribution(encoded, sizes, keys, opened, False, 0.)
    return actions, distribution.logits


@pytest.mark.parametrize("encoding,q_learning", _DEVICE_ENCODINGS)
def test_dynamic_encoder_and_action_features_follow_model_dtype_and_keep_finite_gradients(encoding, q_learning):
    model = _device_policy(encoding, q_learning).to(dtype=torch.float64)
    domains = dict(a=(0, 2), b=(0, 1, 2), c=(1, 2))
    features = model.domain_features(domains)
    assert features.dtype == torch.float64 and features.device == model.dimensions.device
    actions, scores = _device_scores(model, domains)
    assert actions and scores.dtype == torch.float64 and scores.device == model.dimensions.device
    scores.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device and CUDA-enabled PyTorch required")
@pytest.mark.parametrize("encoding,q_learning", _DEVICE_ENCODINGS)
def test_cuda_encoders_train_construct_and_reload_as_cpu_policies(tmp_path, encoding, q_learning):
    from copy import deepcopy

    torch.manual_seed(41)
    cpu = _device_policy(encoding, q_learning).eval()
    gpu = deepcopy(cpu).to("cuda").eval()
    domains = dict(a=(0, 2), b=(0, 1, 2), c=(1, 2))
    cpu_actions, expected = _device_scores(cpu, domains)
    gpu_actions, actual = _device_scores(gpu, domains)
    assert gpu.domain_features(domains).is_cuda and actual.is_cuda
    assert cpu_actions == gpu_actions
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-4)
    gpu.train()
    optimizer = torch.optim.Adam(gpu.parameters(), lr=.001)
    optimizer.zero_grad()
    actual.square().mean().backward()
    gradients = [parameter.grad for parameter in gpu.parameters() if parameter.grad is not None]
    assert gradients and all(gradient.is_cuda and torch.isfinite(gradient).all() for gradient in gradients)
    optimizer.step()
    gpu.eval()
    candidate = next(gpu.proposals(domains, max_candidates=2))
    assert gpu.candidate_trace(candidate, domains)
    path = tmp_path / "cuda.pt"
    torch.save(dict(version=1, kind=gpu.kind, network=gpu.network, config=gpu.config,
                    reward_colors=None, weights=gpu.state_dict(), beta=0.), path)
    restored = runtime.load(path)
    assert all(parameter.device.type == "cpu" for parameter in restored.parameters())
    assert all(buffer.device.type == "cpu" for buffer in restored.buffers())
    actions, expected = _device_scores(gpu, domains)
    restored_actions, actual = _device_scores(restored, domains)
    assert restored_actions == actions
    torch.testing.assert_close(actual, expected.cpu(), atol=1e-4, rtol=1e-4)
