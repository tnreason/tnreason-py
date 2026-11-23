"""Replay training-only heuristic successes to label directly predicted clusters.

An anchor is a variable actually pruned by contraction and subsequent GAC,
including pruning outside the cluster. Contradictions are counted separately.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .candidate_evaluation_mac import prepare_candidate
from .candidate_evaluation_tnreason import SummaryClusterCandidate
from .library import _training, _write
from .library_evaluation import LibraryEvaluator
from .runtime import apply_candidate


_EVALUATOR = None


def _init_worker(network, reward_colors):
    global _EVALUATOR
    _EVALUATOR = LibraryEvaluator(dict(network=network, reward_colors=reward_colors, clusters=[]))


def replay(evaluator, masks, trial, config):
    """Return pre-application masks and exact targets; restore the entire state."""
    raw, state = evaluator.network, evaluator.state
    keys, colors = list(raw.coresDict), evaluator.colors
    selected, outputs = trial["cores"], trial["open"]
    if (not selected or any(type(i) is not int or not 0 <= i < len(keys) for i in selected)
            or any(type(i) is not int or not 0 <= i < len(colors) for i in outputs)
            or len(set(selected)) != len(selected) or len(set(outputs)) != len(outputs)):
        raise ValueError("Invalid teacher cluster indices")
    scope = {color for i in selected for color in raw.coresDict[keys[i]].colors}
    if not {colors[i] for i in outputs} <= scope:
        raise ValueError("Teacher outputs must occur in the selected cores")
    candidate = SummaryClusterCandidate("mask_teacher", tuple(keys[i] for i in selected),
                                        tuple(colors[i] for i in outputs))
    with evaluator.closed(dict(masks=masks)):
        before = state.snapshot_domains()
        prepared = prepare_candidate(state, candidate, **{name: config[name]
            for name in ("max_work", "max_size", "compact_domains") if name in config})
        result = apply_candidate(state, candidate, prepared)
        removed = np.asarray([len(before[color]) - len(state.domains[color]) for color in colors])
        primary_removed = sum(removed[i] for i, color in enumerate(colors) if color in evaluator.primary)
        if (result.removed_values != int(removed.sum())
                or result.removed_values != trial["removed_values"]
                or bool(result.contradiction) != trial["contradiction"]
                or primary_removed != trial["primary_removed_values"]):
            raise ValueError("Replayed teacher outcome differs from the recorded application")
        if result.contradiction:
            return None
        if not np.any(removed > 0):
            raise ValueError("Expected a useful noncontradictory teacher")
        allowed = np.zeros((len(colors), max(state.dimensions.values())), dtype=bool)
        for i, color in enumerate(colors):
            allowed[i, list(before[color])] = True
        cores, opened = np.zeros(len(keys), dtype=bool), np.zeros(len(colors), dtype=bool)
        cores[selected], opened[outputs] = True, True
        return allowed, cores, opened, removed > 0


def _replay_job(job):
    path, expected_hash, puzzle, indices, config = job
    value = Path(path).read_bytes()
    if sha256(value).hexdigest() != expected_hash:
        raise ValueError("Collection row changed after provenance checks")
    row = json.loads(value)
    samples, contradictions = [], 0
    for index in indices:
        trial = row["applications"][index]
        result = replay(_EVALUATOR, row["states"][trial["state"]], trial, config)
        if result is None:
            contradictions += 1
        else:
            samples.append(result)
    if _EVALUATOR.state.decision_level or _EVALUATOR.state._temporary_core_keys:
        raise AssertionError("Teacher replay leaked a decision level or summary")
    return puzzle, samples, contradictions


def _sources(data, paths, limit):
    _, training = _training(data)
    source_hash = sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()
    used, sources, jobs, selected_total, useful_total = set(), [], [], 0, 0
    for directory in map(Path, paths):
        protocol_bytes, library_bytes = ((directory / name).read_bytes()
                                         for name in ("protocol.json", "library.json"))
        protocol, library = json.loads(protocol_bytes), json.loads(library_bytes)
        indices = protocol.get("train_indices", [])
        if (protocol.get("version") != 1 or protocol.get("kind") != "heuristic_cluster_library"
                or not indices or any(type(i) is not int for i in indices)
                or len(set(indices)) != len(indices) or not set(indices) <= set(training)
                or used & set(indices)):
            raise ValueError("Collections must contain disjoint, declared training puzzles")
        if (protocol.get("source_sha256") != source_hash or protocol.get("network") != data["network"]
                or protocol.get("reward_colors") != data.get("reward_colors", data["network"]["colors"])
                or any(library.get(key) != value for key, value in protocol.items())):
            raise ValueError("Collection source, schema or completion metadata differs")
        if not protocol.get("code_sha256") or any(not isinstance(value, str) or len(value) != 64
                for value in protocol["code_sha256"].values()):
            raise ValueError("Missing collector code provenance")
        used.update(indices)
        rows, source_useful = [], 0
        for puzzle in indices:
            path = directory / "puzzles" / f"{puzzle:06d}.json"
            value = path.read_bytes()
            row, row_hash = json.loads(value), sha256(value).hexdigest()
            if row.get("puzzle") != puzzle:
                raise ValueError("Collection row has a different puzzle index")
            useful = [i for i, trial in enumerate(row["applications"])
                      if trial["removed_values"] or trial["contradiction"]]
            if row["statistics"]["useful"] != len(useful):
                raise ValueError("Incomplete successful-application records")
            for trial in row["applications"]:
                if type(trial["state"]) is not int or not 0 <= trial["state"] < len(row["states"]):
                    raise ValueError("Application references a missing domain state")
            source_useful += len(useful)
            chosen = useful[:max(0, limit - selected_total)] if limit is not None else useful
            selected_total += len(chosen)
            rows.append(dict(puzzle=puzzle, sha256=row_hash, useful=len(useful), selected=len(chosen)))
            if chosen:
                jobs.append((str(path), row_hash, puzzle, chosen, protocol["config"]))
        if (library.get("summary", {}).get("puzzles") != len(indices)
                or library["summary"].get("recorded_useful") != source_useful):
            raise ValueError("Completed collection summary differs from its rows")
        useful_total += source_useful
        sources.append(dict(path=str(directory), protocol_sha256=sha256(protocol_bytes).hexdigest(),
            library_sha256=sha256(library_bytes).hexdigest(), train_indices=indices,
            collector_code_sha256=protocol["code_sha256"], config=protocol["config"], rows=rows))
    return source_hash, sorted(used), sources, jobs, useful_total, selected_total


def build(collections, output, *, data, workers=4, limit=None, verbose=False):
    """Create a fresh supervised corpus; --limit bounds globally ordered replays."""
    if (not collections or type(workers) is not int or workers < 1
            or limit is not None and (type(limit) is not int or limit < 1)):
        raise ValueError("Expected positive workers and optional replay limit")
    output, data_path = Path(output), Path(data)
    if output.exists():
        raise FileExistsError(output)
    data_bytes = data_path.read_bytes()
    dataset = json.loads(data_bytes)
    source_hash, training, sources, jobs, useful, selected = _sources(dataset, collections, limit)
    raw, reward = dataset["network"], dataset.get("reward_colors", dataset["network"]["colors"])
    files = [Path(__file__), Path(__file__).with_name("library.py"), Path(__file__).with_name("library_evaluation.py"),
             Path(__file__).with_name("runtime.py"), Path(__file__).with_name("candidate_evaluation_mac.py"),
             Path(__file__).with_name("contraction_cost.py")]
    files.extend(Path(__file__).parent.parent / "constraint_networks" / name
                 for name in ("mac.py", "constraint_networks.py", "generalized_arc_consistency.py"))
    protocol = dict(version=1, kind="supervised_cluster_masks", data_path=str(data_path),
        data_sha256=sha256(data_bytes).hexdigest(), source_sha256=source_hash,
        network=raw, reward_colors=reward, train_indices=training, sources=sources,
        code_sha256={path.name: sha256(path.read_bytes()).hexdigest() for path in files},
        selection_order="Collection argument order, saved training-index order, application order",
        limit=limit, anchor="Any variable whose domain actually shrank after contraction and GAC; "
            "not restricted to cluster outputs or scope; contradictions excluded")
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "protocol.json", protocol)
    arrays, puzzles, contradictions, processed = [[], [], [], []], [], 0, 0
    executor = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                  initargs=(raw, reward)) if workers > 1 else None
    try:
        if executor is None:
            _init_worker(raw, reward)
        results = executor.map(_replay_job, jobs) if executor else map(_replay_job, jobs)
        for number, (puzzle, samples, failures) in enumerate(results, 1):
            contradictions += failures
            processed += len(samples) + failures
            for sample in samples:
                for values, target in zip(arrays, sample):
                    values.append(target)
                puzzles.append(puzzle)
            if verbose:
                print(f"replayed {processed}/{selected} successes in {number}/{len(jobs)} puzzles; "
                      f"{len(puzzles)} pruning examples, {contradictions} contradictions", flush=True)
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    shapes = [(len(raw["colors"]), max(raw["dimensions"])), (len(raw["cores"]),),
              (len(raw["colors"]),), (len(raw["colors"]),)]
    corpus = {name: np.stack(values) if values else np.zeros((0, *shape), dtype=bool)
              for name, values, shape in zip(("allowed", "cores", "outputs", "pruned"), arrays, shapes)}
    corpus.update(puzzles=np.asarray(puzzles, dtype=np.int64), network_json=np.str_(json.dumps(raw)),
                  reward_colors_json=np.str_(json.dumps(reward)))
    with (output / "corpus.npz").open("xb") as stream:
        np.savez_compressed(stream, **corpus)
    summary = dict(available_successes=useful, replayed_successes=processed, samples=len(puzzles),
        contradictions_excluded=contradictions, source_puzzles=len(training),
        sampled_puzzles=len(set(puzzles)), variables=len(raw["colors"]), cores=len(raw["cores"]),
        samples_pruning_outside_outputs=int(np.sum(np.any(corpus["pruned"] & ~corpus["outputs"], axis=1))),
        corpus_sha256=sha256((output / "corpus.npz").read_bytes()).hexdigest())
    _write(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collections", nargs="+", type=Path)
    parser.add_argument("--data", type=Path, required=True, help="Original dataset with explicit training split")
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwritten")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Maximum successful applications replayed, including contradictions")
    args = vars(parser.parse_args())
    print(json.dumps(build(**args, verbose=True)), flush=True)


if __name__ == "__main__":
    main()
