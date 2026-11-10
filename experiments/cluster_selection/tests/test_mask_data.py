from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest

from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from experiments.cluster_selection.data import network_data
from experiments.cluster_selection.library_evaluation import LibraryEvaluator
from experiments.cluster_selection.mask_data import build, replay
from experiments.cluster_selection.runtime import apply_candidate


def _fixture(contradiction=False):
    if contradiction:
        cores = {key: NumpyCore(values=~np.eye(3, dtype=bool), colors=list(key))
                 for key in ("ab", "bc", "ca")}
        selected, outputs = tuple(cores), ()
    else:
        relation = np.array([[1, 0, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
        cores = {"ab": NumpyCore(values=relation, colors=["a", "b"]),
                 "ac": NumpyCore(values=relation, colors=["a", "c"]),
                 "bc": NumpyCore(values=~np.eye(3, dtype=bool), colors=["b", "c"]),
                 "ad": NumpyCore(values=np.eye(3, dtype=bool), colors=["a", "d"])}
        selected, outputs = ("ab", "ac", "bc"), ("a",)
    cores["e"] = NumpyCore(values=np.ones(2, dtype=bool), colors=["e"])
    raw = network_data(ConstraintNetwork(cores, maintain_indices=True))
    colors, keys = raw["colors"], [core["key"] for core in raw["cores"]]
    data = dict(version=1, network=raw, reward_colors=colors,
        initial_domains=[{**{color: [0, 1] for color in colors if color != "e"}, "e": [value]}
                         for value in (0, 1)], train_indices=[0], validation_indices=[1])
    evaluator = LibraryEvaluator(dict(network=raw, reward_colors=colors, clusters=[]))
    state = evaluator.state
    state.push_level()
    for color, values in data["initial_domains"][0].items():
        state.restrict_domain(color, values)
    assert state.propagate()
    before = state.snapshot_domains()
    masks = [sum(1 << value for value in before[color]) for color in colors]
    result = apply_candidate(state, SummaryClusterCandidate("teacher", selected, outputs))
    trial = dict(state=0, cores=[keys.index(key) for key in selected],
        open=[colors.index(color) for color in outputs], removed_values=result.removed_values,
        primary_removed_values=sum(len(before[c]) - len(state.domains[c]) for c in colors),
        contradiction=bool(result.contradiction), level=0)
    state.pop_level()
    return data, evaluator, masks, trial


def _source(tmp_path, contradiction=False):
    data, evaluator, masks, trial = _fixture(contradiction)
    directory = tmp_path / "collection"
    (directory / "puzzles").mkdir(parents=True)
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps(data))
    protocol = dict(version=1, kind="heuristic_cluster_library", network=data["network"],
        reward_colors=data["reward_colors"], train_indices=[0],
        source_sha256=sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest(),
        code_sha256={"collector.py": "0" * 64},
        config=dict(max_work=1_000_000, max_size=100_000, compact_domains=False))
    library = dict(**protocol, summary=dict(puzzles=1, recorded_useful=1))
    row = dict(puzzle=0, states=[masks], applications=[trial], statistics=dict(useful=1))
    for path, value in ((directory / "protocol.json", protocol), (directory / "library.json", library),
                        (directory / "puzzles" / "000000.json", row)):
        path.write_text(json.dumps(value))
    return data_path, directory, data, evaluator, masks, trial


def _snapshot(state):
    return (state.snapshot_domains(), state.statistics, tuple(state._trail), state.decision_level,
            tuple(state._cores), frozenset(state._added_core_keys), tuple(state._temporary_core_keys),
            tuple(state._queue), frozenset(state._queued), state.contradiction,
            tuple(tuple(state._domain_tensors[c]) for c in state.dimensions))


def test_replay_labels_actual_pruning_outside_cluster_and_restores_state():
    data, evaluator, masks, trial = _fixture()
    before = _snapshot(evaluator.state)
    allowed, selected, outputs, pruned = replay(evaluator, masks, trial, {})
    colors = data["network"]["colors"]
    assert [colors[i] for i in np.flatnonzero(pruned)] == ["a", "d"]
    assert outputs[colors.index("a")] and not outputs[colors.index("d")]
    assert not selected[[core["key"] for core in data["network"]["cores"]].index("ad")]
    assert allowed[colors.index("a")].tolist() == [True, True, False]
    assert _snapshot(evaluator.state) == before
    again = replay(evaluator, masks, trial, {})
    assert all(np.array_equal(left, right) for left, right in zip(again, (allowed, selected, outputs, pruned)))
    assert _snapshot(evaluator.state) == before


def test_contradiction_is_excluded_without_fabricating_anchor_labels():
    _, evaluator, masks, trial = _fixture(contradiction=True)
    before = _snapshot(evaluator.state)
    assert trial["contradiction"] and replay(evaluator, masks, trial, {}) is None
    assert _snapshot(evaluator.state) == before


@pytest.mark.parametrize("field,value", [("removed_values", 0), ("primary_removed_values", 0),
                                         ("contradiction", True)])
def test_replay_rejects_changed_outcomes_and_restores_state(field, value):
    _, evaluator, masks, trial = _fixture()
    before = _snapshot(evaluator.state)
    trial[field] = value
    with pytest.raises(ValueError, match="outcome differs"):
        replay(evaluator, masks, trial, {})
    assert _snapshot(evaluator.state) == before


def test_build_exports_exact_pre_application_targets_and_parallel_parity(tmp_path):
    data_path, directory, data, _, _, _ = _source(tmp_path)
    outputs = [tmp_path / "single", tmp_path / "parallel"]
    summaries = [build([directory], output, data=data_path, workers=workers)
                 for workers, output in zip((1, 2), outputs)]
    assert summaries[0] == summaries[1]
    assert summaries[0]["samples"] == summaries[0]["samples_pruning_outside_outputs"] == 1
    with np.load(outputs[0] / "corpus.npz", allow_pickle=False) as first, \
            np.load(outputs[1] / "corpus.npz", allow_pickle=False) as second:
        assert set(first.files) == {"allowed", "cores", "outputs", "pruned", "puzzles",
                                   "network_json", "reward_colors_json"}
        assert all(np.array_equal(first[key], second[key]) for key in first.files)
        assert json.loads(str(first["network_json"])) == data["network"]
        assert first["puzzles"].tolist() == [0]
        assert all(first[name].dtype == np.bool_ for name in ("allowed", "cores", "outputs", "pruned"))
    protocol = json.loads((outputs[0] / "protocol.json").read_text())
    assert protocol["train_indices"] == [0] and protocol["sources"][0]["rows"][0]["sha256"]
    with pytest.raises(FileExistsError):
        build([directory], outputs[0], data=data_path)


def test_build_counts_contradictions_and_keeps_empty_array_shapes(tmp_path):
    data_path, directory, data, _, _, _ = _source(tmp_path, contradiction=True)
    output = tmp_path / "corpus"
    summary = build([directory], output, data=data_path, workers=1, limit=1)
    assert summary["samples"] == 0 and summary["contradictions_excluded"] == 1
    with np.load(output / "corpus.npz", allow_pickle=False) as corpus:
        assert corpus["allowed"].shape == (0, len(data["network"]["colors"]), 3)
        assert corpus["cores"].shape == (0, len(data["network"]["cores"]))


def test_source_overlap_and_reserved_puzzles_are_rejected(tmp_path):
    data_path, directory, _, _, _, _ = _source(tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        build([directory, directory], tmp_path / "overlap", data=data_path, workers=1)
    for filename in ("protocol.json", "library.json"):
        value = json.loads((directory / filename).read_text())
        value["train_indices"] = [1]
        (directory / filename).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="training puzzles"):
        build([directory], tmp_path / "reserved", data=data_path, workers=1)


def test_source_hash_and_completion_rows_are_checked(tmp_path):
    data_path, directory, _, _, _, _ = _source(tmp_path)
    protocol_path = directory / "protocol.json"
    original = json.loads(protocol_path.read_text())
    corrupted = deepcopy(original)
    corrupted["source_sha256"] = "1" * 64
    protocol_path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="source, schema"):
        build([directory], tmp_path / "bad_source", data=data_path, workers=1)
    protocol_path.write_text(json.dumps(original))
    row_path = directory / "puzzles" / "000000.json"
    row = json.loads(row_path.read_text())
    row["applications"] = []
    row_path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="Incomplete"):
        build([directory], tmp_path / "incomplete", data=data_path, workers=1)
