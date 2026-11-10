from copy import deepcopy
import json
import numpy as np
import pytest

from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.cluster_selection import runtime
from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from experiments.cluster_selection.data import network_data


def _network():
    relation = np.array([[1, 0, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
    cores = {"ab": NumpyCore(values=relation, colors=["a", "b"]),
             "ac": NumpyCore(values=relation, colors=["a", "c"]),
             "bc": NumpyCore(values=~np.eye(3, dtype=bool), colors=["b", "c"]),
             "ad": NumpyCore(values=np.eye(3, dtype=bool), colors=["a", "d"])}
    rename = dict(zip("abcd", "uvwx"))
    cores.update({"other_" + key: NumpyCore(values=core.values.copy(),
                  colors=[rename[color] for color in core.colors]) for key, core in tuple(cores.items())})
    cores["e"] = NumpyCore(values=np.ones(3, dtype=bool), colors=["e"])
    return ConstraintNetwork(cores, maintain_indices=True)


def _candidates():
    return (SummaryClusterCandidate("first", ("ab", "ac", "bc"), ("a",)),
            SummaryClusterCandidate("second", ("other_ab", "other_ac", "other_bc"), ("u",)),
            SummaryClusterCandidate("noop", ("e",), ("e",)))


def _domains():
    return {color: (0, 1) for color in "abcuvw"}


def _data():
    return dict(version=1, network=network_data(_network()),
                initial_domains=[{**_domains(), "e": values}
                                 for values in ((0, 1, 2), (0, 1), (1, 2), (0, 2))],
                train_indices=[0, 1], validation_indices=[2], test_indices=[3])


def test_heuristic_observer_preserves_solving_and_sees_updated_domains(monkeypatch):
    network, observations = _network(), []

    def observe(state, candidate, estimate, before, result):
        assert not any(key.startswith("summary_") for key in state._cores)
        assert not state._temporary_core_keys
        assert not state._queue and not state._queued
        assert set(before) == set(state.dimensions)
        assert estimate.work > 0
        observations.append((candidate.key, before, state.snapshot_domains(),
                             state.decision_level, result.removed_values))

    states = [network.create_mac_state(_domains()) for _ in range(2)]
    for state, callback in zip(states, (None, observe)):
        runtime.attach_heuristic(state, on_application=callback)
        monkeypatch.setattr(state.on_closure.generator, "generate", lambda *args, **kwargs: _candidates())
    results = [network.solve_mac(state=state) for state in states]
    assert results[0] == results[1] and results[0].solved
    stats = [{key: value for key, value in state.on_closure.statistics.items() if key != "time_s"}
             for state in states]
    assert stats[0] == stats[1]
    assert len(observations) == stats[1]["applied"]
    first, second, noop = observations[:3]
    assert first[0] == "first" and first[1]["a"] == (0, 1) and first[2]["a"] == (1,)
    assert second[0] == "second" and second[1]["a"] == (1,) and second[1]["u"] == (0, 1)
    assert second[2]["u"] == (1,)
    assert noop[0] == "noop" and noop[1] == noop[2] and noop[4] == 0
    assert all(item[3] == 0 for item in observations[:3])
    assert observations[0][1]["a"] == (0, 1)  # Saved domains remain independent of live updates.


def test_heuristic_observer_pruning_retains_parent_backtracking(monkeypatch):
    state = _network().create_mac_state(_domains())
    assert state.propagate()
    before, observations = state.snapshot_domains(), []
    runtime.attach_heuristic(state, on_application=lambda *args: observations.append(args))
    monkeypatch.setattr(state.on_closure.generator, "generate", lambda *args, **kwargs: _candidates())
    state.push_level()
    assert state.propagate()
    assert state.domains["a"] == state.domains["u"] == (1,)
    assert len(observations) == 3 and state.decision_level == 1
    assert all(record[0] is state for record in observations)
    assert not state._temporary_core_keys and not any(key.startswith("summary_") for key in state._cores)
    assert observations[0][3] == before
    state.pop_level()
    assert state.snapshot_domains() == before and not state.contradiction


def test_heuristic_observer_records_contradiction_before_parent_rollback(monkeypatch):
    cores = {key: NumpyCore(values=~np.eye(3, dtype=bool), colors=list(key))
             for key in ("ab", "bc", "ca")}
    state = ConstraintNetwork(cores).create_mac_state({color: (0, 1) for color in "abc"})
    assert state.propagate()
    before, observations = state.snapshot_domains(), []

    def observe(current, candidate, estimate, domains, result):
        assert current is state and current.decision_level == 1
        assert current.contradiction and result.contradiction
        assert set(current._cores) == set(cores) and not current._temporary_core_keys
        observations.append(domains)

    runtime.attach_heuristic(state, on_application=observe)
    candidate = SummaryClusterCandidate("odd", tuple(cores), ())
    monkeypatch.setattr(state.on_closure.generator, "generate", lambda *args, **kwargs: (candidate,))
    state.push_level()
    assert not state.propagate()
    assert observations == [before]
    state.pop_level()
    assert not state.contradiction and state.snapshot_domains() == before


def test_heuristic_observer_excludes_rejected_and_cached_applications(monkeypatch):
    state = _network().create_mac_state(_domains())
    assert state.propagate()
    observations = []
    runtime.attach_heuristic(state, on_application=lambda *args: observations.append(args))
    noop = _candidates()[-1]
    monkeypatch.setattr(state.on_closure.generator, "generate", lambda *args, **kwargs: (noop, noop))
    assert state.propagate()
    assert len(observations) == 1 and state.on_closure.statistics["skipped"] == 1
    rejected = _network().create_mac_state(_domains())
    assert rejected.propagate()
    runtime.attach_heuristic(rejected, max_work=1, on_application=lambda *args: pytest.fail("Rejected trial observed"))
    monkeypatch.setattr(rejected.on_closure.generator, "generate", lambda *args, **kwargs: _candidates()[:1])
    assert rejected.propagate() and rejected.on_closure.statistics["rejected"] == 1


def _mock_generation(monkeypatch):
    def generate(self, domains, **options):
        candidates = _candidates()
        if len(domains["e"]) == 2:
            # Same original cores, distinct outputs; reversed orders must canonicalize.
            alternate = SummaryClusterCandidate("alternate", ("bc", "ac", "ab"), ("b", "a"))
            return (alternate, *candidates)
        return candidates

    monkeypatch.setattr(runtime.ClusterCandidateGenerator, "generate", generate)


@pytest.mark.parametrize("record_failures", [False, True])
def test_library_deduplicates_successes_but_keeps_distinct_outputs(tmp_path, monkeypatch, record_failures):
    from experiments.cluster_selection.library import collect

    data = _data()
    _mock_generation(monkeypatch)
    result = collect(data, tmp_path / "run", nodes=1, seconds=5, record_failures=record_failures)
    assert result["summary"]["puzzles"] == 2
    assert result["summary"]["useful"] == result["summary"]["root_successes"] == 4
    assert result["summary"]["dfs_successes"] == 0
    assert result["summary"]["distinct_clusters"] == 3
    assert result["train_indices"] == [0, 1]
    assert result["network"] == data["network"]
    core_idx = {core["key"]: idx for idx, core in enumerate(data["network"]["cores"])}
    color_idx = {color: idx for idx, color in enumerate(data["network"]["colors"])}
    triangle = sorted(core_idx[key] for key in ("ab", "ac", "bc"))
    entries = [entry for entry in result["clusters"] if entry["cores"] == triangle]
    assert {tuple(entry["open"]) for entry in entries} == {(color_idx["a"],),
                                                             tuple(sorted((color_idx["a"], color_idx["b"])))}
    common = next(entry for entry in result["clusters"] if entry["cores"] != triangle)
    assert common["successes"] == common["root_successes"] == 2 and common["puzzles"] == [0, 1]
    assert all(entry["successes"] > 0 for entry in result["clusters"])
    assert result["growth"][-1]["clusters"] == 3
    records = [json.loads(path.read_text()) for path in sorted((tmp_path / "run" / "puzzles").glob("*.json"))]
    assert {row["puzzle"] for row in records} == {0, 1}
    trials = [trial for row in records for trial in row["applications"]]
    assert any(not (trial["removed_values"] or trial["contradiction"]) for trial in trials) == record_failures
    for row in records:
        assert len(row["states"][0]) == len(data["network"]["colors"])
        assert row["states"][0][color_idx["a"]] == 0b11
        assert all(0 <= trial["state"] < len(row["states"]) for trial in row["applications"])
        first_pruned = {item["variable"]: item["removed"] for item in row["applications"][0]["pruned_values"]}
        assert first_pruned[color_idx["a"]] == first_pruned[color_idx["d"]] == [0]
        assert all(sum(len(item["removed"]) for item in trial["pruned_values"]) == trial["removed_values"]
                   for trial in row["applications"])
    assert json.loads((tmp_path / "run" / "library.json").read_text()) == result


@pytest.mark.parametrize("bad_split", ["missing", "indices", "fingerprints"])
def test_library_rejects_training_holdout_leakage_before_writing(tmp_path, bad_split):
    from experiments.cluster_selection.library import collect

    data = deepcopy(_data())
    if bad_split == "missing":
        del data["train_indices"]
    elif bad_split == "indices":
        data["validation_indices"] = [0]
    else:
        # Index 3 remains reserved even if absent from named validation/test lists.
        data["test_indices"] = []
        data["initial_domains"][3] = {**data["initial_domains"][0], "e": [2, 0, 1, 1]}
    with pytest.raises(ValueError, match="training|reserved"):
        collect(data, tmp_path / "run", nodes=1, seconds=5)
    assert not (tmp_path / "run").exists()


def test_library_records_success_discovered_only_after_branching(tmp_path, monkeypatch):
    from experiments.cluster_selection.library import collect

    cores = _network().coresDict.copy()
    for color in "abc":
        cores["z" + color] = NumpyCore(values=np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
                                      colors=["z", color])
    network = ConstraintNetwork(cores, maintain_indices=True)
    data = dict(version=1, network=network_data(network), initial_domains=[{}], train_indices=[0])
    monkeypatch.setattr(runtime.ClusterCandidateGenerator, "generate", lambda *args, **kwargs: _candidates()[:1])
    output = tmp_path / "run"
    result = collect(data, output, nodes=32, seconds=5, record_failures=True)
    assert result["summary"]["statuses"] == {"solved": 1}
    assert result["summary"]["root_successes"] == 0 and result["summary"]["dfs_successes"] == 1
    assert result["clusters"][0]["dfs_successes"] == 1
    row = json.loads((output / "puzzles" / "000000.json").read_text())
    assert row["applications"][0]["level"] == 0 and row["applications"][0]["removed_values"] == 0
    useful = [trial for trial in row["applications"] if trial["removed_values"] or trial["contradiction"]]
    assert len(useful) == 1 and useful[0]["level"] == 1 and useful[0]["removed_values"] == 2


def test_library_resume_reuses_committed_puzzles_and_rejects_changed_protocol(tmp_path, monkeypatch):
    from experiments.cluster_selection import library

    _mock_generation(monkeypatch)
    data, output, calls = _data(), tmp_path / "run", []
    run = library._run

    def interrupted(*args):
        calls.append(args[2])
        if len(calls) == 2:
            raise RuntimeError("interrupted between puzzle commits")
        return run(*args)

    monkeypatch.setattr(library, "_run", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        library.collect(data, output, nodes=1, seconds=5)
    committed = next((output / "puzzles").glob("*.json"))
    saved = committed.read_bytes()
    assert len(list((output / "puzzles").glob("*.json"))) == 1
    assert not (output / "library.json").exists()
    with pytest.raises(ValueError, match="identical"):
        library.collect(data, output, nodes=2, seconds=5, resume=True)

    resumed = []

    def resume(*args):
        resumed.append(args[2])
        return run(*args)

    monkeypatch.setattr(library, "_run", resume)
    result = library.collect(data, output, nodes=1, seconds=5, resume=True)
    assert resumed == [calls[1]] and committed.read_bytes() == saved
    assert result["summary"]["puzzles"] == 2 and result["summary"]["distinct_clusters"] == 3
    with pytest.raises(ValueError, match="complete"):
        library.collect(data, output, nodes=1, seconds=5, resume=True)


def test_parallel_collection_matches_serial_heuristic_successes(tmp_path):
    from experiments.cluster_selection.library import collect

    kwargs = dict(nodes=4, seconds=10, allowance=8, max_cores=4, max_open_colors=2,
                  record_failures=True)
    serial = collect(_data(), tmp_path / "serial", workers=1, **kwargs)
    parallel = collect(_data(), tmp_path / "parallel", workers=2, **kwargs)
    assert serial["clusters"] and serial["clusters"] == parallel["clusters"]
    assert serial["summary"] == parallel["summary"] and serial["growth"] == parallel["growth"]
    for puzzle in serial["train_indices"]:
        rows = [json.loads((tmp_path / mode / "puzzles" / f"{puzzle:06d}.json").read_text())
                for mode in ("serial", "parallel")]
        assert all(rows[0][key] == rows[1][key] for key in ("status", "nodes", "states", "applications"))
        assert {k: v for k, v in rows[0]["statistics"].items() if k != "time_s"} == {
            k: v for k, v in rows[1]["statistics"].items() if k != "time_s"}
