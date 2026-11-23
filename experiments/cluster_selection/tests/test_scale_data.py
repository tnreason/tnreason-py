"""Current-GAC filtering, split preservation and reusable-state correctness."""
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import signal
import shlex
import shutil
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tnreason.engine.workload_to_numpy import NumpyCore

from experiments.cluster_selection.data import network_data
from experiments.cluster_selection.reinforcement import scale_data
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.sudoku_tests.standard_constraints import get_sudoku_constraint_network


SOLUTION = "123456789456789123789123456214365897365897214897214365531642978642978531978531642"


@pytest.fixture(scope="module")
def dataset():
    network = ConstraintNetwork(get_sudoku_constraint_network(), maintain_indices=True)
    return dict(version=1, network=network_data(network),
                reward_colors=[key for key in network.dimension_dict if key.startswith("pos_")],
                puzzles=[SOLUTION, SOLUTION[:2] + "." * 79, SOLUTION[:3] + "." * 78],
                train_indices=[0], validation_indices=[1, 2],
                samples=[dict(puzzle=0, domains=[])])


def _write(path, puzzles, difficulty=1):
    pq.write_table(pa.Table.from_pylist([dict(puzzle=board.replace(".", "0"), solution=SOLUTION,
                                           difficulty=difficulty) for board in puzzles]), path,
                   row_group_size=2)


def test_reused_gac_state_filters_solved_and_restores_every_puzzle(dataset):
    scale_data._initialize(dataset)
    before = scale_data._STATE.snapshot_domains()
    rows = [dict(puzzle=SOLUTION, solution=SOLUTION),
            dict(puzzle="." * 81, solution=SOLUTION),
            dict(puzzle="9" + "." * 80, solution=SOLUTION)]
    assert [status for status, _ in scale_data._check(rows)] == ["solved", "unresolved", "invalid_solution"]
    assert scale_data._STATE.snapshot_domains() == before
    assert scale_data._STATE.decision_level == 0
    assert not scale_data._STATE.contradiction
    assert tuple(scale_data._STATE._cores) == tuple(core["key"] for core in dataset["network"]["cores"])


def test_filter_preserves_network_indices_and_excludes_reserved_duplicates(tmp_path, dataset):
    before = deepcopy(dataset)
    first, reserved = "1" + "." * 80, "." * 81
    second = ".2" + "." * 79
    _write(tmp_path / "train_0.parquet", [SOLUTION, first, first, reserved, second])
    result = scale_data.prepare(dataset, tmp_path, target=2, workers=1, batch_size=8,
                               reserved=[reserved], drop_validation=(1,), seed=3)
    assert dataset == before
    assert result["network"] == before["network"]
    assert result["puzzles"][:3] == before["puzzles"]
    assert result["train_indices"] == [0, 1, 3, 4]
    assert result["validation_indices"] == [2]
    assert set(result["puzzles"][3:]) == {first, second}
    assert "samples" not in result
    metadata = result["scale_preparation"]
    assert metadata["target_reached"]
    assert metadata["existing_unresolved_train_puzzles"] == 1
    assert metadata["unresolved_train_puzzles"] == 3
    assert metadata["stats"]["retained"] == metadata["stats"]["unresolved"] == 2
    assert metadata["stats"]["excluded"] == 1
    assert metadata["stats"]["duplicate"] == 2
    assert [record["puzzle_index"] for record in metadata["records"]] == [3, 4]


def test_source_priority_does_not_replace_gac_check_and_limit_is_explicit(tmp_path, dataset):
    # The source's higher difficulty label is a scan-order hint, not the filter.
    _write(tmp_path / "train_0.parquet", ["." * 81], difficulty=0)
    _write(tmp_path / "train_1.parquet", [SOLUTION[:-1] + "."], difficulty=1)
    result = scale_data.prepare(dataset, tmp_path, target=1, max_scanned=1,
                               workers=1, batch_size=1, drop_validation=())
    metadata = result["scale_preparation"]
    assert metadata["source_files"][0]["path"].endswith("train_1.parquet")
    assert metadata["stats"]["checked"] == metadata["stats"]["solved"] == 1
    assert not metadata["target_reached"] and metadata["stats"]["retained"] == 0


def test_source_validation_reserves_normalized_exact_grids(tmp_path):
    _write(tmp_path / "valid_0.parquet", ["." * 81, "1" + "." * 80])
    assert scale_data.reserve_validation(tmp_path) == {"." * 81, "1" + "." * 80}


def _binary_network(unique=False):
    cores = {"ab": NumpyCore(values=np.eye(2, dtype=bool), colors=["a", "b"])}
    if unique:
        cores.update(bc=NumpyCore(values=np.eye(2, dtype=bool), colors=["b", "c"]),
                     ac=NumpyCore(values=np.array([[1, 1], [1, 0]], dtype=bool), colors=["a", "c"]))
    return ConstraintNetwork(cores, maintain_indices=True)


def test_generic_uniqueness_certifies_unsat_blocking_and_validates_both_solutions():
    unique = _binary_network(unique=True)
    before = {key: core.values.tobytes() for key, core in unique.coresDict.items()}
    result = scale_data.check_uniqueness(unique, {}, dict(a=0, b=0, c=0), nodes=100, seconds=2.)
    assert result["status"] == "unique"
    assert result["reference_status"] == "solved" and result["alternative_status"] == "unsat"
    assert before == {key: core.values.tobytes() for key, core in unique.coresDict.items()}
    multiple = scale_data.check_uniqueness(_binary_network(), {}, dict(a=0, b=0), nodes=100, seconds=2.)
    assert multiple["status"] == "multiple" and multiple["alternative"] == dict(a=1, b=1)
    invalid = scale_data.check_uniqueness(unique, {}, dict(a=1, b=1, c=1), nodes=100, seconds=2.)
    assert invalid["status"] == "invalid_reference" and invalid["reference_status"] == "unsat"
    unsat = ConstraintNetwork({"a": NumpyCore(values=np.zeros(2, dtype=bool), colors=["a"])},
                             maintain_indices=True)
    assert scale_data.check_uniqueness(unsat, {}, dict(a=0))["status"] == "invalid_reference"


def test_uniqueness_on_selected_colors_handles_fixed_references_and_search_limits():
    network = _binary_network()
    fixed = scale_data.check_uniqueness(network, dict(a=(0,)), dict(a=0))
    assert fixed["status"] == "unique"  # Unselected auxiliary values do not define alternatives.
    conflict = scale_data.check_uniqueness(network, dict(a=(1,)), dict(a=0))
    assert conflict["status"] == "invalid_reference"
    limited = scale_data.check_uniqueness(network, {}, dict(a=0, b=0), nodes=1)
    assert limited["status"] == "unknown" and limited["alternative_status"] == "node_limit"
    independent = ConstraintNetwork({c: NumpyCore(values=np.ones(2, dtype=bool), colors=[c])
                                     for c in "ab"}, maintain_indices=True)
    limited_reference = scale_data.check_uniqueness(independent, {}, dict(a=0), nodes=1)
    assert limited_reference["status"] == "unknown" and limited_reference["reference_status"] == "node_limit"
    with pytest.raises(ValueError, match="at least one"):
        scale_data.check_uniqueness(network, {}, {})
    for reference in (dict(missing=0), dict(a=2), dict(a=.5)):
        with pytest.raises(ValueError):
            scale_data.check_uniqueness(network, {}, reference)


def test_uniqueness_timeout_is_unknown_and_restores_alarm_handler(monkeypatch):
    original = ConstraintNetwork.solve_mac
    previous = signal.getsignal(signal.SIGALRM)

    def interrupted(self, *args, **kwargs):
        if len(self.coresDict) > 1:
            signal.raise_signal(signal.SIGALRM)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ConstraintNetwork, "solve_mac", interrupted)
    result = scale_data.check_uniqueness(_binary_network(), {}, dict(a=0, b=0))
    assert result["status"] == "unknown" and result["alternative_status"] == "wall_timeout"
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


def test_blocking_tensors_are_linear_and_do_not_overwrite_existing_names(monkeypatch):
    reference = {f"x{i}": 0 for i in range(32)}
    cores = {color: NumpyCore(values=np.ones(2, dtype=bool), colors=[color]) for color in reference}
    cores["__alternative_end"] = NumpyCore(values=np.array([True, False]), colors=["__alternative_state_0"])
    network = ConstraintNetwork(cores, maintain_indices=True)
    before = {key: core.values.tobytes() for key, core in cores.items()}
    original, captured = ConstraintNetwork.solve_mac, []

    def observed(self, *args, **kwargs):
        added = {key: core for key, core in self.coresDict.items() if key not in cores}
        if added:
            captured.append(added)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ConstraintNetwork, "solve_mac", observed)
    result = scale_data.check_uniqueness(network, {c: (0,) for c in list(reference)[:-1]}, reference)
    assert result["status"] == "multiple" and result["alternative"]["x31"] == 1
    assert len(captured) == 1 and len(captured[0]) == len(reference) + 2
    assert max(len(core.colors) for core in captured[0].values()) == 3
    assert sum(core.values.size for core in captured[0].values()) == 8 * len(reference) + 4
    assert before == {key: core.values.tobytes() for key, core in network.coresDict.items()}


def _unique_preparation_data():
    from experiments.cluster_selection.supervised.collect import sudoku_domains
    reference = {color: values[0] for color, values in sudoku_domains(SOLUTION).items()}
    a, b = list(reference)[:2]
    cores = {color: NumpyCore(values=np.arange(9) == value, colors=[color])
             for color, value in reference.items() if color not in (a, b)}
    left, right, closing = np.zeros((9, 2), dtype=bool), np.zeros((9, 2), dtype=bool), np.zeros((9, 9), dtype=bool)
    left[reference[a], 0] = left[8, 1] = True
    right[reference[b], 0] = right[8, 1] = True
    closing[reference[a], reference[b]] = closing[reference[a], 8] = closing[8, reference[b]] = True
    cores.update(left=NumpyCore(values=left, colors=[a, "link"]),
                 right=NumpyCore(values=right, colors=[b, "link"]),
                 closing=NumpyCore(values=closing, colors=[a, b]))
    network = ConstraintNetwork(cores, maintain_indices=True)
    return dict(version=1, network=network_data(network), reward_colors=list(reference),
                puzzles=[SOLUTION], train_indices=[], validation_indices=[0])


def test_unique_preparation_retains_certified_stalled_puzzles_and_selected_shards(tmp_path):
    data = _unique_preparation_data()
    before = deepcopy(data)
    _write(tmp_path / "train_0.parquet", ["." * 81])
    _write(tmp_path / "train_1.parquet", ["." * 81], difficulty=99)
    result = scale_data.prepare(data, tmp_path, target=1, workers=1, batch_size=1, drop_validation=(),
                               unique_only=True, uniqueness_nodes=100, uniqueness_seconds=2.,
                               shards=["train_0.parquet"])
    metadata = result["scale_preparation"]
    assert data == before and result["network"] == data["network"]
    assert result["puzzles"] == [SOLUTION, "." * 81] and result["train_indices"] == [1]
    assert result["validation_indices"] == [0]
    assert metadata["unique_only"] and metadata["target_reached"]
    assert metadata["stats"]["unique"] == metadata["stats"]["unresolved"] == metadata["stats"]["retained"] == 1
    assert metadata["stats"]["multiple"] == metadata["stats"]["unknown"] == 0
    assert metadata["selected_shards"] == ["train_0.parquet"]
    assert len(metadata["source_files"]) == 1
    checked = metadata["records"][0]["uniqueness"]
    assert checked["status"] == "unique" and checked["reference_colors"] == 81
    assert checked["alternative_status"] == "unsat"
    assert scale_data._STATE.decision_level == 0 and not scale_data._STATE.contradiction
    assert tuple(scale_data._STATE._cores) == tuple(core["key"] for core in data["network"]["cores"])
    with pytest.raises(ValueError, match="Unknown training shards"):
        scale_data.prepare(data, tmp_path, shards=["train_missing.parquet"])


def test_unique_preparation_does_not_keep_unknown_results(tmp_path, monkeypatch):
    data = _unique_preparation_data()
    _write(tmp_path / "train_0.parquet", ["." * 81])
    monkeypatch.setattr(scale_data, "check_uniqueness", lambda *args, **kwargs:
                        dict(status="unknown", reference_status="solved", alternative_status="node_limit"))
    result = scale_data.prepare(data, tmp_path, target=1, workers=1, drop_validation=(), unique_only=True)
    stats = result["scale_preparation"]["stats"]
    assert stats["unknown"] == stats["unresolved"] == 1 and stats["retained"] == 0
    assert result["puzzles"] == data["puzzles"]


def test_parallel_filter_is_deterministic_and_matches_single_worker(tmp_path, dataset):
    _write(tmp_path / "train_0.parquet", ["." * 81, "1" + "." * 80, ".2" + "." * 79])
    single = scale_data.prepare(dataset, tmp_path, target=2, workers=1, batch_size=1,
                                drop_validation=(), seed=5)
    parallel = scale_data.prepare(dataset, tmp_path, target=2, workers=2, batch_size=1,
                                  drop_validation=(), seed=5)
    assert parallel["puzzles"] == single["puzzles"]
    assert parallel["train_indices"] == single["train_indices"]
    assert parallel["scale_preparation"]["records"] == single["scale_preparation"]["records"]


@pytest.mark.parametrize("nested,requested,expected,decoder", [(False, "auto", 8192, None),
                         (True, "auto", 65536, None), (True, "65536", 65536, None),
                         (False, "auto", 8192, "discrepancy"), (False, "auto", 8192, "sampling"), (False, "auto", 8192, "sampling_grow")])
def test_scaled_gpu_launcher_renders_fresh_branches_and_actual_learner_arguments(tmp_path, monkeypatch,
                                                                            nested, requested, expected, decoder):
    from experiments.cluster_selection.reinforcement import learn, qlearn
    repo = Path(__file__).resolve().parents[3]
    source = repo / "experiments/cluster_selection/reinforcement"
    shared = tmp_path / "shared repo/experiments/cluster_selection/reinforcement"
    (shared / "results").mkdir(parents=True)
    for name in ("submit_teacher_branches.sh", "teacher_branches_gpu.sub", "teacher_branches_gpu.dag"):
        shutil.copy2(source / name, shared / name)
    data = tmp_path / "scaled.json"
    data.write_text(json.dumps(dict(scale_preparation=dict(unresolved_train_puzzles=10051))))
    candidates, attempts, seconds = (256, 256, 10800) if nested else (64, 32, 600)
    command = ["bash", str(shared / "submit_teacher_branches.sh"), sys.executable, "--data", str(data),
               "--passes", "10", "--chunk-size", "500", "--seeds", "1", "--max-jobs", "3", "--dry-run"]
    if nested:
        command += ["--nested-teacher", "--teacher-candidates", str(candidates), "--teacher-max-steps", requested,
                    "--cluster-attempts", str(attempts), "--validation-seconds", str(seconds)]
    if decoder is not None:
        command += ["--decoder", decoder]
    expected_decoder = decoder or "depth_first"
    result = subprocess.run(command, capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONPATH": str(repo) + os.pathsep + os.environ.get("PYTHONPATH", "")})
    run = Path(next(line.removeprefix("Run directory: ") for line in result.stdout.splitlines()
                    if line.startswith("Run directory: ")))
    plan = json.loads((run / "run_plan.json").read_text())
    assert plan["initialization"] == "seeded_random" and plan["initial_checkpoint"] is None
    assert plan["dry_run"] and plan["max_jobs"] == 3 and plan["epochs_per_phase"] == 210
    assert plan["teacher_candidates"] == candidates and plan["teacher_max_steps"] == expected
    assert plan["nested_teacher"] == nested and plan["cluster_attempts"] == attempts and plan["validation_seconds"] == seconds
    assert plan["teacher_probes"] == 80 and plan["teacher_mutations"] == 32
    assert plan["decoder"] == plan["common_settings"]["decoder"] == expected_decoder
    stages = ([dict(candidates=n, max_steps=256 * n) for n in (32, 64, 128, 256)] if nested else
              [dict(candidates=64, max_steps=8192)])
    assert plan["teacher_search_stages"] == stages
    assert plan["teacher_search_steps_nominal"] == (122880 if nested else 8192)
    dag = (run / "teacher_branches_gpu.dag").read_text().splitlines()
    assert [line.split()[1] for line in dag if line.startswith("JOB ")] == ["WARMUP", "TEACHER0", "REINFORCE0", "Q0"]
    assert dag[-1] == "PARENT WARMUP CHILD TEACHER0 REINFORCE0 Q0"
    assert len(plan["jobs"]) == 4 and not list(run.glob("*/policy*.pt"))
    template = (run / "teacher_branches_gpu.sub").read_text()
    assert "if $(fresh_start:false)\n    checkpoint_args =\nelse" in template
    arguments = next(line.split(" = ", 1)[1][1:-1] for line in template.splitlines() if line.startswith("arguments = "))
    macro = re.compile(r"\$\(([^():]+)(?::([^()]*))?\)")
    captured = []
    for line, job in zip((line for line in dag if line.startswith("VARS ")), plan["jobs"]):
        values = dict(token.split("=", 1) for token in shlex.split(line)[2:])
        assert values["teacher_max_steps"] == str(expected) and values["cluster_attempts"] == str(attempts)
        assert values["validation_seconds"] == str(seconds) and values["teacher_candidates"] == str(candidates)
        assert values["teacher_search_args"] == ("--nested-teacher" if nested else "")
        assert values["decoder"] == job["decoder"] == expected_decoder
        assert (values["fresh_start"] == "true") == job["fresh_start"] == (job["node"] == "WARMUP")
        checkpoint = None if job["fresh_start"] else str(run / "warmup/policy.best.pt")
        assert job["initial_checkpoint"] == checkpoint
        values.update(run_root=str(run), checkpoint_args="" if checkpoint is None else
                      "--initial-checkpoint " + shlex.quote(checkpoint))
        tokens = shlex.split(macro.sub(lambda match: values.get(match[1], match[2] or ""), arguments))
        learner = qlearn if job["learner"] == "qlearn" else learn
        monkeypatch.setattr(learner, "fit", lambda data, **options: captured.append(options))
        monkeypatch.setattr(sys, "argv", [tokens[2], *tokens[3:]])
        learner.main()
        options = captured[-1]
        assert options["teacher_candidates"] == candidates and options["teacher_max_steps"] == expected
        assert options["teacher_probes"] == 80 and options["teacher_mutations"] == 32
        assert options["nested_teacher"] == nested and options["cluster_attempts"] == attempts
        assert options["decoder"] == expected_decoder
        assert options["validation_seconds"] == seconds and options["initial_checkpoint"] == (Path(checkpoint) if checkpoint else None)
        assert options["output"] == run / job["run_name"] / "policy.pt"
        if learner is learn:
            assert options["warmup"] == (0 if job["node"] == "REINFORCE0" else 210)


@pytest.mark.parametrize("arguments,message", [
    (["--data", "unused", "--teacher-candidates", "0"], "positive integers"),
    (["--data", "unused", "--teacher-max-steps", "0"], "auto or a positive integer"),
    (["--data", "unused", "--cluster-attempts", "-1"], "positive integers"),
    (["--data", "unused", "--validation-seconds", "0"], "positive integers"),
    (["--nested-teacher"], "require --data"),
    (["--decoder", "discrepancy"], "require --data"),
    (["--data", "unused", "--decoder", "unknown"], "depth_first, discrepancy, sampling or sampling_grow"),
])
def test_gpu_launcher_rejects_invalid_budgets_before_creating_a_run(arguments, message):
    script = Path(__file__).resolve().parents[1] / "reinforcement/submit_teacher_branches.sh"
    result = subprocess.run(["bash", str(script), *arguments, "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 2 and message in result.stderr
