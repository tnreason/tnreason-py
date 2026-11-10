"""Train direct cluster masks from replay-verified examples; optionally validate MAC.

Example: python -m experiments.cluster_selection.mask_train CORPUS/corpus.npz
    --output NEW_DIRECTORY --device cuda --epochs 30 --batch-size 32
    --data development.json --validation-limit 8 --validation-interval 5

Training always starts from random weights. Validation uses only the declared
validation split; checkpoints, metrics and baseline results are separate files.
"""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from .mask_model import MaskPolicy
from .reinforcement.experiment import fingerprints, instance_hashes, instances, propagate, solve


def _hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _corpus(path):
    protocol_path = path.with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text())
    summary = json.loads(path.with_name("summary.json").read_text())
    if type(protocol.get("version")) is not int or protocol["version"] != 1 or protocol.get("kind") != "supervised_cluster_masks":
        raise ValueError("Expected a version-1 supervised-cluster-mask corpus protocol")
    if summary.get("corpus_sha256") != _hash(path):
        raise ValueError("Corpus archive differs from its replay summary hash")
    with np.load(path, allow_pickle=False) as archive:
        raw = json.loads(str(archive["network_json"]))
        reward = json.loads(str(archive["reward_colors_json"]))
        arrays = {key: archive[key] for key in ("allowed", "cores", "outputs", "pruned", "puzzles")}
    n, v, c = len(arrays["puzzles"]), len(raw["colors"]), len(raw["cores"])
    if protocol.get("network") != raw or protocol.get("reward_colors") != reward:
        raise ValueError("Corpus network/reward schema differs from its protocol")
    train = protocol.get("train_indices", ())
    if (not train or any(type(i) is not int or i < 0 for i in train)
            or len(set(train)) != len(train) or set(arrays["puzzles"].tolist()) - set(train)):
        raise ValueError("Corpus puzzle IDs must belong to declared training indices")
    if summary.get("samples") != n:
        raise ValueError("Corpus sample count differs from its replay summary")
    shapes = dict(allowed=(n, v, max(raw["dimensions"])), cores=(n, c), outputs=(n, v), pruned=(n, v))
    if not n or any(arrays[key].dtype != np.bool_ or arrays[key].shape != shape for key, shape in shapes.items()):
        raise ValueError("Expected nonempty boolean mask arrays matching the raw network")
    if arrays["puzzles"].ndim != 1 or arrays["puzzles"].dtype.kind not in "iu":
        raise ValueError("Expected one integer puzzle index per example")
    if (not arrays["allowed"].any(2).all() or not arrays["cores"].any(1).all()
            or not arrays["outputs"].any(1).all() or not arrays["pruned"].any(1).all()
            or set(reward) - set(raw["colors"])):
        raise ValueError("Each example needs nonempty domains, cores, outputs and actual pruned variables")
    return raw, reward, arrays, protocol, protocol_path


def _validation(path, indices, limit, raw, puzzles, protocol):
    """Use this corpus's source dataset and only its reserved validation rows."""
    data = json.loads(path.read_text())
    if data["network"] != raw or _hash(path) != protocol.get("data_sha256"):
        raise ValueError("Validation data must match the corpus source hash and network")
    declared = data.get("validation_indices", ())
    chosen = list(declared if indices is None else indices)
    if (not chosen or len(set(chosen)) != len(chosen) or any(type(i) is not int for i in chosen)
            or set(chosen) - set(declared) or set(chosen) & set(data.get("test_indices", ()))):
        raise ValueError("Use distinct indices from the declared validation split; test rows are excluded")
    train = set(data.get("train_indices", ())) | set(protocol.get("train_indices", ())) | set(puzzles.tolist())
    if set(chosen) & train:
        raise ValueError("Validation indices overlap training")
    initial, _ = instances(data)
    if any(i < 0 or i >= len(initial) for i in (*chosen, *train)):
        raise ValueError("Training or validation index outside source dataset")
    if fingerprints(data, [initial[i] for i in chosen]) & fingerprints(data, [initial[i] for i in train]):
        raise ValueError("Validation initial domains overlap training")
    if limit:
        chosen = chosen[:limit]
    network = ConstraintNetwork({core["key"]: NumpyCore(
        values=np.asarray(core["values"], dtype=bool).reshape(core["shape"]),
        colors=[raw["colors"][i] for i in core["colors"]]) for core in raw["cores"]}, maintain_indices=True)
    exposure = sorted(instance_hashes(data, [initial[i] for i in set(puzzles.tolist())]))
    return network, initial, chosen, exposure


def _evaluate(model, context, modes, nodes, seconds, allowance):
    network, initial, indices, _ = context
    results = {}
    for mode in modes:
        rows = []
        print(f"validating {mode}: {len(indices)} puzzles, allowance {allowance}", flush=True)
        for position, index in enumerate(indices, 1):
            options = dict(seconds=seconds, baseline_candidates=allowance,
                           baseline_steps=max(8192, 256 * allowance),
                           cluster_attempts=allowance, stop_on_noop=True)
            root = propagate(network, initial[index], mode, model, **options)
            search = solve(network, initial[index], mode, model, nodes=nodes, **options)
            if root["status"] == "invalid" or search["status"] == "invalid":
                raise AssertionError("Cluster evaluation produced an invalid assignment")
            rows.append(dict(puzzle=index, root=root, solve=search))
            if position == 1 or position % 5 == 0 or position == len(indices):
                print(f"validation {mode} {position}/{len(indices)}: puzzle {index}, "
                      f"{search['status']}, DFS nodes {search['nodes']}", flush=True)
        resolved = [row for row in rows if row["solve"]["status"] in ("solved", "unsat")]
        stats = [row["root"]["cluster_statistics"] for row in rows]
        results[mode] = dict(instances=len(rows), resolved=len(resolved),
            solved=sum(row["solve"]["solved"] for row in rows),
            total_nodes=sum(row["solve"]["nodes"] for row in resolved) if len(resolved) == len(rows) else None,
            completed_nodes=sum(row["solve"]["nodes"] for row in resolved),
            branch_free=sum(row["solve"]["solved"] and row["solve"]["nodes"] == 1 for row in rows),
            root_solved=sum(row["root"]["solved"] for row in rows),
            root_useful=sum(row.get("useful", 0) for row in stats),
            root_attempts=sum(row.get("selected", row.get("applied", 0)) for row in stats),
            mean_root_progress=float(sum(row["root"].get("progress") or 0. for row in rows) / len(rows)), rows=rows)
    return results


def _save(path, model, training, epoch, metrics):
    weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    with path.open("xb") as stream:
        torch.save(dict(version=1, kind="mask", network=model.network, config=model.config,
                        reward_colors=model.reward_colors, weights=weights,
                        training={**training, "epoch": epoch}, metrics=metrics), stream)


def fit(corpus, output, *, epochs=30, batch_size=32, lr=.0003, seed=0, device="cpu",
        width=32, rounds=3, anchor_rounds=3, max_cores=64, max_open_colors=6,
        max_work=1_000_000, max_size=100_000, allowance=64, data=None,
        validation_indices=None, validation_limit=0, validation_interval=5,
        nodes=10_000, seconds=120., threads=1):
    """Batched supervised learning, with actual greedy cluster/MAC validation."""
    corpus, output = Path(corpus), Path(output)
    if (min(epochs, batch_size, width, rounds, anchor_rounds, max_cores, max_open_colors,
            max_work, max_size, allowance, validation_interval, nodes, threads) < 1
            or validation_limit < 0 or not isfinite(lr) or lr <= 0 or not isfinite(seconds) or seconds <= 0):
        raise ValueError("Training and evaluation budgets must be positive")
    training_device = torch.device(device)
    if training_device.type not in ("cpu", "cuda") or training_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Choose cpu or an available CUDA device")
    raw, reward, arrays, protocol, protocol_path = _corpus(corpus)
    if (arrays["cores"].sum(1).max() > max_cores or arrays["outputs"].sum(1).max() > max_open_colors):
        raise ValueError("Training targets exceed the configured core/output limits")
    if data is None and validation_indices is not None:
        raise ValueError("Validation indices require --data")
    context = _validation(Path(data), validation_indices, validation_limit, raw, arrays["puzzles"], protocol) if data else None
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = MaskPolicy(raw, width=width, rounds=rounds, anchor_rounds=anchor_rounds,
                       max_cores=max_cores, max_open_colors=max_open_colors, max_work=max_work,
                       max_size=max_size, cluster_attempts=allowance).to(training_device)
    model.reward_colors = reward
    tensors = {key: torch.from_numpy(value) for key, value in arrays.items() if key != "puzzles"}
    training = dict(method="direct_mask_imitation", seed=seed, from_scratch=True,
                    corpus=str(corpus.resolve()), corpus_sha256=_hash(corpus),
                    protocol_sha256=_hash(protocol_path),
                    corpus_summary_sha256=_hash(corpus.with_name("summary.json")),
                    train_puzzles=sorted(set(arrays["puzzles"].tolist())))
    if context:
        training["train_instance_hashes"] = context[3]
    manifest = dict(version=1, **training, source_protocol=protocol, config=model.config,
        epochs=epochs, batch_size=batch_size, lr=lr, device=str(training_device),
        validation_device="cpu", validation_indices=context[2] if context else [],
        validation_interval=validation_interval, node_budget=nodes, seconds_per_run=seconds,
        allowance=allowance, baseline_steps=max(8192, 256 * allowance), stop_on_noop=True, total_work=None,
        code_sha256={str(path.resolve()): _hash(path) for path in
                     (Path(__file__), Path(__file__).with_name("mask_model.py"),
                      Path(__file__).with_name("encoder.py"), Path(__file__).with_name("runtime.py"))})
    _write(output / "manifest.json", manifest)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    history, best, selected = [], None, "initial.pt"
    print(f"Direct cluster-mask imitation on {training_device}: {len(arrays['puzzles'])} examples; from random weights", flush=True)
    if training_device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(training_device)}; PyTorch {torch.__version__}; validation on CPU", flush=True)

    def validation(epoch):
        if context is None:
            return None
        validator = deepcopy(model).cpu().eval() if training_device.type != "cpu" else model.eval()
        modes = ("plain", "heuristic", "trained") if epoch == 0 else ("trained",)
        result = _evaluate(validator, context, modes, nodes, seconds, allowance)
        _write(output / f"validation_{epoch:04d}.json", result)
        if epoch == 0:
            _write(output / "baselines.json", {key: result[key] for key in ("plain", "heuristic")})
        return result["trained"]

    _save(output / "initial.pt", model, training, 0, None)
    initial = validation(0)
    if initial is not None:
        best = (initial["resolved"], -initial["completed_nodes"], initial["branch_free"], initial["mean_root_progress"])
        print(f"initial validation: resolved {initial['resolved']}/{initial['instances']}, DFS nodes {initial['total_nodes']}, root useful {initial['root_useful']}/{initial['root_attempts']}", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        began, totals = perf_counter(), {}
        order = torch.randperm(len(arrays["puzzles"]), generator=generator)
        for indices in order.split(batch_size):
            batch = {key: value[indices].to(training_device) for key, value in tensors.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = model.loss(batch["allowed"], batch["cores"], batch["outputs"], batch["pruned"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            for key, value in {**metrics, "loss": loss.detach()}.items():
                number = float(value.detach()) if torch.is_tensor(value) else float(value)
                if not isfinite(number):
                    raise ValueError(f"Nonfinite training metric: {key}")
                totals[key] = totals.get(key, 0.) + len(indices) * number
        row = dict(epoch=epoch, training_seconds=perf_counter() - began,
                   metrics={key: value / len(order) for key, value in totals.items()})
        result = validation(epoch) if epoch % validation_interval == 0 or epoch == epochs else None
        checkpoint = f"epoch_{epoch:04d}.pt"
        if result is not None:
            row["validation"] = {key: value for key, value in result.items() if key != "rows"}
            rank = (result["resolved"], -result["completed_nodes"], result["branch_free"], result["mean_root_progress"])
            if best is None or rank > best:
                best, selected = rank, checkpoint
        elif context is None:
            selected = checkpoint
        history.append(row)
        _save(output / checkpoint, model, training, epoch, row)
        _write(output / f"metrics_{epoch:04d}.json", row)
        text = ", ".join(f"{key} {value:.4f}" for key, value in row["metrics"].items())
        if result is not None:
            text += f", validation DFS nodes {result['total_nodes']}, root useful {result['root_useful']}/{result['root_attempts']}"
        print(f"epoch {epoch}/{epochs}: {text}", flush=True)
    selection = dict(checkpoint=selected, rule="validation resolved count, DFS nodes, branch-free solves, root progress"
        if context else "final epoch; no validation supplied", validation_indices=context[2] if context else [],
        selection_metrics=best, epochs=epochs, no_test_evaluation=True)
    _write(output / "selection.json", selection)
    return model.eval(), history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data", type=Path, help="Original corpus source; enables validation, never test evaluation")
    parser.add_argument("--validation-indices", type=int, nargs="+")
    for key, default in dict(epochs=30, batch_size=32, lr=.0003, seed=0, width=32, rounds=3,
                            anchor_rounds=3, max_cores=64, max_open_colors=6, max_work=1_000_000,
                            max_size=100_000, allowance=64, validation_limit=0,
                            validation_interval=5, nodes=10_000, seconds=120., threads=1).items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(default), default=default)
    fit(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
