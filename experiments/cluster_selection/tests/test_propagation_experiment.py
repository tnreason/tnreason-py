"""Full-root evaluation must measure propagation rather than hidden branching."""
import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.cluster_selection import runtime
from experiments.cluster_selection.data import network_data
from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from experiments.cluster_selection.reinforcement.experiment import First, evaluate_roots, propagate, solve


def _network():
    return ConstraintNetwork({
        "ab": NumpyCore(values=np.eye(2, dtype=bool), colors=["a", "b"]),
        "bc": NumpyCore(values=np.eye(2, dtype=bool), colors=["b", "c"]),
        "ac": NumpyCore(values=np.array([[0, 1], [1, 1]], dtype=bool), colors=["a", "c"]),
    }, maintain_indices=True)


def _model(network, candidate=None):
    return SimpleNamespace(kind="construction", network=network_data(network), reward_colors=list("abc"),
                           config=dict(max_cores=10, max_open_colors=4, max_work=10000,
                                       max_size=1000, heuristic_candidates=0),
                           propose=lambda domains: candidate)


def _candidate():
    return SummaryClusterCandidate("triangle", ("ab", "bc", "ac"), ("a",))


def test_contraction_solves_arc_consistent_root_without_branching(monkeypatch):
    network = _network()
    model = _model(network, _candidate())
    monkeypatch.setattr(network, "solve_mac", lambda *args, **kwargs: pytest.fail("No DFS allowed"))
    plain = propagate(network, {}, "plain", model, seconds=2)
    learned = propagate(network, {}, "trained", model, seconds=2)
    assert plain["status"] == "stalled" and plain["nodes"] == 0
    assert learned["status"] == "solved" and learned["solved"] and learned["valid"]
    assert learned["assignment"] == dict(a=1, b=1, c=1)
    assert learned["nodes"] == 0 and not learned["baseline_solved"]
    assert learned["progress"] == 1 and learned["cluster_removed_values"] == 3
    assert learned["work"]["contractions"] == learned["cluster_statistics"]["useful"] == 1
    assert set(network.coresDict) == {"ab", "bc", "ac"}


def test_cumulative_work_limit_prevents_contraction_and_is_reported():
    network = _network()
    row = propagate(network, {}, "trained", _model(network, _candidate()), seconds=2, total_work=1)
    assert row["status"] == "work_limit" and not row["solved"]
    assert row["work"]["budget_exhausted"] and row["work"]["contractions"] == 0
    assert row["domains"] == dict(a=(0, 1), b=(0, 1), c=(0, 1))
    assert row["progress"] == row["cluster_removed_values"] == 0


def test_root_metrics_exclude_easy_gac_roots_from_incremental_solved_rate():
    network = _network()
    rows = evaluate_roots(network, [{}, {"a": (1,)}], _model(network, _candidate()), seconds=2)
    assert rows["instances"] == rows["solved"] == 2
    assert rows["baseline_solved"] == rows["unresolved"] == rows["additional_solved"] == 1
    assert rows["additional_solved_fraction"] == rows["mean_progress"] == 1


def test_contradictory_root_is_resolved_but_never_counted_as_solution():
    network = ConstraintNetwork({key: NumpyCore(values=~np.eye(2, dtype=bool), colors=list(key))
                                 for key in ("ab", "bc", "ac")}, maintain_indices=True)
    model = _model(network, _candidate())
    report = evaluate_roots(network, [{}], model, seconds=2)
    row = report["rows"][0]
    assert row["status"] == "unsat" and not row["solved"] and row["progress"] is None
    assert report["contradictions"] == 1 and report["solved_fraction"] == 0
    assert row["resolved"] and not row["baseline_contradiction"]
    assert report["resolved_fraction"] == report["additional_resolved_fraction"] == 1


def test_scalar_contradiction_with_fixed_variables_is_not_an_easy_solved_root():
    network = ConstraintNetwork({"a": NumpyCore(values=np.array([1, 0]), colors=["a"]),
                                 "false": NumpyCore(values=np.array(0), colors=[])}, maintain_indices=True)
    model = SimpleNamespace(reward_colors=["a"])
    report = evaluate_roots(network, [{"a": (0,)}], model, seconds=2, mode="plain")
    assert report["baseline_solved"] == report["solved"] == 0
    assert report["contradictions"] == 1
    assert report["rows"][0]["progress"] is None
    assert report["rows"][0]["baseline_contradiction"]
    assert report["resolved_fraction"] == 1 and report["additional_resolved_fraction"] == 0
    assert report["unresolved"] == report["additional_resolved"] == 0


def test_only_cluster_proved_unsat_counts_as_additionally_resolved():
    network = ConstraintNetwork({key: NumpyCore(values=~np.eye(2, dtype=bool), colors=list(key))
                                 for key in ("ab", "bc", "ac")}, maintain_indices=True)
    report = evaluate_roots(network, [{"a": (0,)}, {}], _model(network, _candidate()), seconds=2)
    assert report["resolved"] == 2 and report["baseline_contradictions"] == 1
    assert report["unresolved"] == report["additional_resolved"] == 1
    assert report["additional_resolved_fraction"] == 1 and report["additional_solved_fraction"] == 0


def test_timeout_restores_contraction_function_and_returns_partial_domains():
    network = _network()
    model = _model(network)
    original = runtime.contract_candidate

    def timeout(domains):
        raise TimeoutError("injected")

    model.propose = timeout
    row = propagate(network, {}, "trained", model, seconds=2)
    assert row["status"] == "wall_timeout" and not row["solved"]
    assert row["domains"] == dict(a=(0, 1), b=(0, 1), c=(0, 1))
    assert runtime.contract_candidate is original


def test_structural_baselines_remain_available_when_training_proposals_disabled():
    network = _network()
    model = _model(network)
    first = First(model, candidates=4)
    assert first.propose(dict(a=(0, 1), b=(0, 1), c=(0, 1))) is not None
    row = propagate(network, {}, "heuristic", model, seconds=2, baseline_candidates=4)
    assert row["solved"] and row["work"]["contractions"] >= 1


def test_full_search_uses_identical_cluster_accounting_and_validates_solution():
    network = _network()
    model = _model(network, _candidate())
    plain = solve(network, {}, "plain", model, nodes=20, seconds=2)
    learned = solve(network, {}, "trained", model, nodes=20, seconds=2)
    assert plain["solved"] and learned["solved"] and plain["valid"] and learned["valid"]
    assert learned["nodes"] < plain["nodes"]
    assert learned["work"]["contractions"] == 1


@pytest.mark.parametrize("seconds,total_work", [(0, None), (float("nan"), None), (1, 0)])
def test_invalid_budgets_fail_before_evaluation(seconds, total_work):
    network = _network()
    with pytest.raises(ValueError, match="positive"):
        propagate(network, {}, "plain", _model(network), seconds=seconds, total_work=total_work)


@pytest.mark.parametrize("recorded_exposure", [True, False])
def test_cli_checkpoint_exposure_prevents_relabelling_promoted_training_instance(
        tmp_path, monkeypatch, capsys, recorded_exposure):
    from experiments.cluster_selection.reinforcement import experiment
    from experiments.cluster_selection.reinforcement.model import Policy

    raw = network_data(_network())
    evaluation = dict(network=raw, reward_colors=list("abc"),
                      initial_domains=[dict(a=[0, 1], b=[0, 1], c=[0, 1])], instance_ids=["claimed-held-out"])
    old_training = {**evaluation, "initial_domains": [dict(a=[1])], "instance_ids": ["old-corpus"]}
    # This full-domain instance was subsequently promoted into training. Normalizing
    # duplicates and ordering must identify it despite its new evaluation label.
    promoted = [dict(c=[1, 0, 1], a=[1, 0], b=[0, 1, 0])]
    assert not experiment.fingerprints(evaluation, evaluation["initial_domains"]) & experiment.fingerprints(
        old_training, old_training["initial_domains"])
    model = Policy(raw, width=8, rounds=1, max_cores=3)
    checkpoint = dict(version=1, kind="construction", network=raw, config=model.config,
                      reward_colors=list("abc"), beta=0., weights=model.state_dict())
    if recorded_exposure:
        checkpoint["training"] = dict(train_instance_hashes=sorted(experiment.instance_hashes(evaluation, promoted)))
    checkpoint_path = tmp_path / "policy.pt"
    torch.save(checkpoint, checkpoint_path)
    evaluation_path, training_path = tmp_path / "evaluation.json", tmp_path / "old-training.json"
    evaluation_path.write_text(json.dumps(evaluation))
    training_path.write_text(json.dumps(old_training))
    output = tmp_path / "results.jsonl"
    monkeypatch.setattr(sys, "argv", ["experiment", str(evaluation_path), "--training-data", str(training_path),
        "--checkpoint", str(checkpoint_path), "--output", str(output), "--indices", "0", "--modes", "plain",
        "--nodes", "20", "--seconds", "2"])
    if recorded_exposure:
        with pytest.raises(SystemExit) as stopped:
            experiment.main()
        assert stopped.value.code == 2
        assert "overlap checkpoint training" in capsys.readouterr().err
        assert not output.exists()
    else:
        experiment.main()
        manifest, solved = [json.loads(line) for line in output.read_text().splitlines()]
        assert manifest["exact_training_overlap"] == manifest["checkpoint_training_overlap"] == 0
        assert manifest["checkpoint_training_fingerprints_available"] is False
        assert solved["solved"] and solved["valid"]
