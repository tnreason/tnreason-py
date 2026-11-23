"""Direct-mask training provenance and actual solver-validation checks."""
from hashlib import sha256
import json

import numpy as np
import pytest
import torch

from experiments.cluster_selection import runtime
from experiments.cluster_selection.mask_train import fit


def _inputs(tmp_path):
    network = dict(colors=["a", "b", "c"], dimensions=[2, 2, 2], cores=[
        dict(key="ab", colors=[0, 1], shape=[2, 2], values=[1, 0, 0, 1]),
        dict(key="bc", colors=[1, 2], shape=[2, 2], values=[1, 0, 0, 1]),
        dict(key="ca", colors=[2, 0], shape=[2, 2], values=[1, 1, 1, 0])])
    data = dict(network=network, reward_colors=network["colors"],
                initial_domains=[{}, {"a": [0]}, {"a": [1]}],
                train_indices=[0], validation_indices=[1], test_indices=[2])
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps(data))
    corpus = tmp_path / "corpus.npz"
    np.savez_compressed(corpus, allowed=np.ones((2, 3, 2), dtype=bool),
        cores=np.ones((2, 3), dtype=bool), outputs=np.array([[1, 0, 0]] * 2, dtype=bool),
        pruned=np.ones((2, 3), dtype=bool), puzzles=np.zeros(2, dtype=np.int64),
        network_json=np.array(json.dumps(network)), reward_colors_json=np.array(json.dumps(network["colors"])))
    (tmp_path / "protocol.json").write_text(json.dumps(dict(version=1, kind="supervised_cluster_masks",
        network=network, reward_colors=network["colors"], train_indices=[0],
        data_sha256=sha256(data_path.read_bytes()).hexdigest())))
    (tmp_path / "summary.json").write_text(json.dumps(dict(samples=2,
        corpus_sha256=sha256(corpus.read_bytes()).hexdigest())))
    return corpus, data_path


def test_batched_training_saves_random_initial_and_distinct_epochs(tmp_path, monkeypatch):
    corpus, data = _inputs(tmp_path)
    output = tmp_path / "run"
    attach, structural_steps = runtime.attach_heuristic, []
    def observed(state, **options):
        structural_steps.append(options["max_steps"])
        return attach(state, **options)
    monkeypatch.setattr(runtime, "attach_heuristic", observed)
    _, history = fit(corpus, output, epochs=1, batch_size=2, width=8, rounds=1,
        anchor_rounds=1, max_cores=3, max_open_colors=1, data=data,
        validation_interval=1, allowance=64, seconds=10., nodes=20)
    initial = torch.load(output / "initial.pt", weights_only=True)
    trained = torch.load(output / "epoch_0001.pt", weights_only=True)
    assert initial["training"]["epoch"] == 0 and initial["training"]["from_scratch"]
    assert trained["training"]["epoch"] == 1 and trained["kind"] == "mask"
    assert any(not torch.equal(value, trained["weights"][key]) for key, value in initial["weights"].items())
    assert np.isfinite(history[0]["metrics"]["loss"])
    result = json.loads((output / "validation_0001.json").read_text())["trained"]
    assert result["resolved"] == 1 and result["total_nodes"] == 1
    assert result["rows"][0]["solve"]["valid"]
    assert set(json.loads((output / "baselines.json").read_text())) == {"plain", "heuristic"}
    assert structural_steps == [16384, 16384]
    assert json.loads((output / "selection.json").read_text())["no_test_evaluation"]
    assert runtime.load(output / "epoch_0001.pt").network == trained["network"]
    marker = output / "keep.txt"
    marker.write_text("unchanged")
    with pytest.raises(FileExistsError):
        fit(corpus, output, epochs=1)
    assert marker.read_text() == "unchanged"


@pytest.mark.parametrize("indices", ([0], [2], [1, 1]))
def test_validation_rejects_training_test_and_repeated_rows(tmp_path, indices):
    corpus, data = _inputs(tmp_path)
    with pytest.raises(ValueError, match="declared validation|overlap"):
        fit(corpus, tmp_path / "run", epochs=1, data=data, validation_indices=indices)
    assert not (tmp_path / "run").exists()


def test_validation_requires_unchanged_source_hash(tmp_path):
    corpus, data = _inputs(tmp_path)
    data.write_text(data.read_text() + "\n")
    with pytest.raises(ValueError, match="source hash"):
        fit(corpus, tmp_path / "run", epochs=1, data=data)


@pytest.mark.parametrize("field,value", [
    ("version", 2), ("kind", "unverified"), ("network", {}),
    ("reward_colors", ["a"]), ("train_indices", [1])])
def test_rejects_tampered_corpus_protocol(tmp_path, field, value):
    corpus, _ = _inputs(tmp_path)
    path = tmp_path / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol[field] = value
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="protocol|schema|training"):
        fit(corpus, tmp_path / "run", epochs=1)
    assert not (tmp_path / "run").exists()


def test_rejects_changed_archive_before_training(tmp_path):
    corpus, _ = _inputs(tmp_path)
    with corpus.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="summary hash"):
        fit(corpus, tmp_path / "run", epochs=1)


def test_checkpoint_with_unsolved_root_validation_loads_safely(tmp_path):
    from tnreason.engine.workload_to_numpy import NumpyCore
    from experiments.constraint_networks.constraint_networks import ConstraintNetwork
    from experiments.cluster_selection.data import network_data
    from experiments.cluster_selection.mask_model import MaskPolicy
    from experiments.cluster_selection.mask_train import _evaluate, _save

    network = ConstraintNetwork({"free": NumpyCore(values=np.ones(2), colors=["a"])}, maintain_indices=True)
    model = MaskPolicy(network_data(network), width=8, rounds=1, max_cores=1, max_open_colors=1)
    model.reward_colors = ["a"]
    result = _evaluate(model, (network, [{}], [0], []), ("trained",), 10, 10., 1)["trained"]
    assert not result["rows"][0]["root"]["baseline_solved"]
    path = tmp_path / "validated.pt"
    metrics = {key: value for key, value in result.items() if key != "rows"}
    _save(path, model, {}, 1, metrics)
    assert runtime.load(path).network == model.network
