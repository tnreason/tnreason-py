import json
import subprocess
import sys
from copy import deepcopy
import numpy as np
import pytest
from experiments.cluster_selection import collect as collector
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.sudoku_tests.standard_constraints import get_sudoku_constraint_network
from tnreason.engine.workload_to_numpy import NumpyCore

torch = pytest.importorskip("torch")
from experiments.cluster_selection.model import Predictor, load
from experiments.cluster_selection.train import evaluate, fit


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _network():
    relation = np.array([[1, 0, 1], [1, 1, 1], [1, 1, 1]])
    cores = {"ab": NumpyCore(values=relation, colors=["a", "b"]),
             "ac": NumpyCore(values=relation, colors=["a", "c"]),
             "bc": NumpyCore(values=~np.eye(3, dtype=bool), colors=["b", "c"]),
             "ad": NumpyCore(values=np.eye(3, dtype=bool), colors=["a", "d"])}
    return ConstraintNetwork(cores, maintain_indices=True)


def _collect(network=None):
    return collector.collect(_network() if network is None else network,
        [{"a": (0, 1), "b": pair, "c": pair} for pair in ((0, 1), (0, 2))],
        max_states=2, states_per_puzzle=1, max_candidates=8)


def _snapshot(state):
    return (state.snapshot_domains(), state.statistics, state.decision_level,
            tuple(state._trail), tuple(state._level_markers), tuple(state._cores),
            frozenset(state._added_core_keys), frozenset(state._changed_colors),
            tuple(state._queue), frozenset(state._queued), state.on_closure,
            tuple((id(value), tuple(value)) for value in state._domain_tensors.values()))


def test_collection_preserves_each_trial_and_stores_static_data_once(monkeypatch):
    original, calls = collector.evaluate_candidate, []
    network = _network()
    before = {key: core.values.copy() for key, core in network.coresDict.items()}

    def observed(state, candidate, **kwargs):
        snapshot = _snapshot(state)
        result = original(state, candidate, **kwargs)
        assert _snapshot(state) == snapshot
        calls.append(result)
        return result

    monkeypatch.setattr(collector, "evaluate_candidate", observed)
    data = json.loads(json.dumps(_collect(network)))
    assert len(data["samples"]) == 2 and len(calls) == 16
    assert any(result.removed_domain_values > 0 for result in calls)
    assert any(result.removed_domain_values == 0 for result in calls)
    assert all("network" not in sample and len(sample["domains"]) == 4 for sample in data["samples"])
    for key, values in before.items():
        np.testing.assert_array_equal(values, network.coresDict[key].values)


def test_real_sudoku_collection_visits_search_levels_without_extra_cores():
    network = ConstraintNetwork(get_sudoku_constraint_network(2), maintain_indices=True)
    data = collector.collect(network, [{}], reward_colors=[c for c in network.dimension_dict if c.startswith("pos_")],
                             max_states=3, states_per_puzzle=3, max_nodes=12, max_candidates=2)
    assert len(data["samples"]) == 3
    assert any(sample["level"] > 0 for sample in data["samples"])
    assert len(data["network"]["cores"]) == len(network.coresDict) == 256
    assert len(data["samples"][0]["domains"]) == 128


@pytest.mark.parametrize("message", ["Candidate exceeds the contraction work or tensor-size budget", "malformed"])
def test_only_budget_rejections_are_skipped(monkeypatch, message):
    def fail(*args, **kwargs):
        raise ValueError(message)

    monkeypatch.setattr(collector, "evaluate_candidate", fail)
    if message == "malformed":
        with pytest.raises(ValueError, match="malformed"):
            _collect()
    else:
        data = _collect()
        assert not data["samples"] and data["rejected"] > 0


def test_domain_masks_and_tensor_axes_are_preserved():
    values = np.array([[1, 0, 1], [0, 1, 0]])
    network = ConstraintNetwork({"xy": NumpyCore(values=values.T, colors=["y", "x"])}, maintain_indices=True)
    raw = collector.network_data(network)
    model = Predictor(raw, width=8, rounds=1)
    assert raw["cores"][0]["shape"] == [3, 2]
    assert raw["cores"][0]["values"] == values.T.astype(bool).ravel().tolist()
    torch.testing.assert_close(model.core_features[0, :6], torch.tensor(values.T.ravel(), dtype=torch.float32))
    assert model.variable_idx.tolist() == [0, 1]
    torch.testing.assert_close(model.axes, torch.eye(2))
    first = model.domain_features([[0, 1], [0, 1]])
    second = model.domain_features([[1, 2], [0, 1]])
    assert not torch.equal(first, second) and first[0, -3] == second[0, -3]
    with pytest.raises(ValueError, match="dimension"):
        model.domain_features([[3], [0]])
    for value in (.5, True):
        with pytest.raises(ValueError, match="dimension"):
            model.domain_features([[value], [0]])


def test_candidates_share_encoding_and_all_message_paths_receive_gradients(monkeypatch):
    data = _collect()
    sample = data["samples"][0]
    model = Predictor(data["network"], width=8, rounds=2)
    original, calls = model.encode, []

    def encode(domains):
        calls.append(1)
        return original(domains)

    monkeypatch.setattr(model, "encode", encode)
    prediction = model(sample["domains"], sample["candidates"])
    assert prediction.shape == (8, 3) and len(calls) == 1
    prediction.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    reordered = deepcopy(sample["candidates"])
    for candidate in reordered:
        candidate["cores"].reverse()
        candidate["open"].reverse()
    torch.testing.assert_close(model(sample["domains"], reordered), prediction, atol=1e-5, rtol=1e-5)


def test_skip_action_and_empty_candidate_lists():
    data = _collect()
    sample = data["samples"][0]
    model = Predictor(data["network"], width=8, rounds=1)
    with torch.no_grad():
        for parameter in model.head.parameters():
            parameter.zero_()
    assert model.select(sample["domains"], sample["candidates"]) is None
    assert model.select(sample["domains"], []) is None
    assert model(sample["domains"], []).shape == (0, 3)
    with torch.no_grad():
        model.head[-1].bias[0] = 1
    assert model.select(sample["domains"], sample["candidates"]) == 0


def test_candidate_local_messages_distinguish_equal_sized_topologies(monkeypatch):
    scopes = [("x", "a"), ("x", "b"), ("x", "c"), ("x", "a"), ("a", "b"), ("b", "c")]
    network = ConstraintNetwork({str(i): NumpyCore(values=np.ones((2, 2)), colors=list(scope))
                                 for i, scope in enumerate(scopes)}, maintain_indices=True)
    model = Predictor(collector.network_data(network), width=1, rounds=1)
    monkeypatch.setattr(model, "encode", lambda domains: (torch.zeros(4, 1), torch.zeros(6, 1)))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.to_variable[-1].bias.fill_(1)
        model.update_variable[0].weight[0, 1] = 1
        model.update_variable[0].bias.fill_(-1.5)
        model.update_variable[-1].weight.fill_(1)
        model.head[0].weight[0, 2] = 1
        model.head[2].weight[0, 0] = 1
        model.head[4].weight[0, 0] = 1
    candidates = [dict(cores=keys, open=[0, 1, 2, 3], estimate=[100, 16, 16])
                  for keys in ([0, 1, 2], [3, 4, 5])]
    prediction = model([[0, 1]] * 4, candidates)
    torch.testing.assert_close(prediction[:, 0], torch.tensor([1.5, 1.0]))


def test_tiny_training_checkpoint_and_offline_evaluation(tmp_path):
    data = _collect()
    model, history = fit(data, epochs=80, width=16, rounds=2, batch_size=2, lr=.005)
    assert history[-1] < history[0] / 2
    path = tmp_path / "predictor.pt"
    torch.save(dict(network=model.network, config=model.config, weights=model.state_dict(),
                    reward_colors=model.reward_colors, version=1), path)
    restored = load(path)
    for sample in data["samples"]:
        torch.testing.assert_close(restored(sample["domains"], sample["candidates"]),
                                   model(sample["domains"], sample["candidates"]))
    result = evaluate(restored, data)
    assert result["states"] == 2 and result["mean_regret"] >= -1e-9
    incompatible = deepcopy(data)
    incompatible["network"]["cores"][0]["values"][0] = False
    with pytest.raises(ValueError, match="schema"):
        evaluate(restored, incompatible)
    incompatible = deepcopy(data)
    incompatible["reward_colors"] = ["b"]
    with pytest.raises(ValueError, match="reward"):
        evaluate(restored, incompatible)


def test_collection_is_torch_free_and_cli_round_trip(tmp_path):
    clean = subprocess.run([sys.executable, "-B", "-c",
        "import sys; import experiments.cluster_selection.collect; assert 'torch' not in sys.modules"],
        capture_output=True, text=True, timeout=30)
    assert clean.returncode == 0, clean.stderr
    puzzles, dataset, checkpoint = (tmp_path / name for name in ("puzzles.txt", "data.json", "model.pt"))
    puzzles.write_text("0" * 16 + "\n")
    commands = [
        ["collect", str(puzzles), "--num", "2", "--states", "2", "--candidates", "2", "--output", str(dataset)],
        ["train", str(dataset), "--epochs", "2", "--width", "8", "--rounds", "1", "--output", str(checkpoint)],
        ["train", str(dataset), "--checkpoint", str(checkpoint)],
    ]
    for module, *args in commands:
        result = subprocess.run([sys.executable, "-B", "-m", f"experiments.cluster_selection.{module}", *args],
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
    assert dataset.exists() and checkpoint.exists()
    assert "mean_regret" in result.stdout and "Offline" in result.stdout


def test_invalid_limits_and_boards():
    with pytest.raises(ValueError, match="positive"):
        collector.collect(_network(), [{}], max_states=0)
    with pytest.raises(ValueError, match="grid"):
        collector.sudoku_domains("x" * 81)
    assert collector.sudoku_domains("1" + "." * 15, 2) == {"pos_0_0_0_0": (0,)}
