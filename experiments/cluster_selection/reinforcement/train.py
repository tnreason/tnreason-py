"""Legacy treewidth-penalty REINFORCE; use learn.py for current actor training.

The state/closure/checkpoint helpers here are shared by the current learners.
Keep this trainer for reproducing earlier experiments and checkpoints.
"""
import argparse
from contextlib import contextmanager
import json
from math import isfinite, sqrt
from pathlib import Path
from random import Random
from tempfile import NamedTemporaryFile
from time import perf_counter
import numpy as np
import torch
from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from ..candidate_evaluation_mac import evaluate_candidate
from ..contraction_cost import estimate_treewidth
from ..data import network_data
from .model import Policy, load


COST_METRIC = "treewidth_min_fill_in"


def _state(data):
    if data.get("version") != 1 or not data.get("samples"):
        raise ValueError("Expected a version-1 dataset with closed domain states")
    raw = data["network"]
    network = ConstraintNetwork({core["key"]: NumpyCore(
        values=np.asarray(core["values"], dtype=bool).reshape(core["shape"]),
        colors=[raw["colors"][idx] for idx in core["colors"]]) for core in raw["cores"]}, maintain_indices=True)
    if network_data(network) != raw or set(data["reward_colors"]) - set(raw["colors"]):
        raise ValueError("Invalid network schema or reward colors")
    state = network.create_mac_state()
    if not state.propagate():
        raise ValueError("Expected a noncontradictory base network")
    return state


@contextmanager
def _closed(state, sample, colors):
    if len(sample["domains"]) != len(colors):
        raise ValueError("Expected every variable's domain")
    domains = {color: tuple(values) for color, values in zip(colors, sample["domains"])}
    state.push_level()
    try:
        for color, values in domains.items():
            state.restrict_domain(color, values)
        if not state.propagate() or state.snapshot_domains() != domains:
            raise ValueError("Expected noncontradictory GAC-closed domain states")
        yield state
    finally:
        state.pop_level()


def trial(model, state, *, cost_weight=1.0, contradiction_reward=1.0, greedy=False):
    """Reward pruning minus estimated treewidth; seconds are reported only."""
    started = perf_counter()
    candidate, log_prob, entropy, _ = model.rollout(state.domains, greedy=greedy)
    gain, contradiction, rejected, treewidth = 0, False, False, 0
    if candidate is not None:
        treewidth = estimate_treewidth(state._cores, candidate)
        try:
            result = evaluate_candidate(state, candidate, reward_colors=model.reward_colors,
                                        max_work=model.config["max_work"], max_size=model.config["max_size"])
            gain, contradiction = result.removed_domain_values, result.contradiction
        except ValueError as error:
            if str(error) not in ("Candidate exceeds the contraction work or tensor-size budget",
                                  "NumPy contraction supports at most 52 distinct colors"):
                raise
            rejected = True
    reward = gain + contradiction_reward * contradiction - cost_weight * treewidth
    metrics = dict(reward=reward, gain=gain, treewidth=treewidth, cost_s=perf_counter() - started,
                   selected=int(candidate is not None),
                   useful=int(gain > 0 or contradiction), contradictions=int(contradiction), rejected=int(rejected))
    return log_prob, entropy, metrics


def _weights(cost_weight, contradiction_reward):
    if any(not isfinite(value) or value < 0 for value in (cost_weight, contradiction_reward)):
        raise ValueError("Reward weights must be finite and nonnegative")


def _save(path, model, history, cost_weight, contradiction_reward, selection=None, *,
          cost_metric=COST_METRIC, **metadata):
    checkpoint = dict(version=1, kind=model.kind, network=model.network, config=model.config,
                      reward_colors=model.reward_colors, weights=model.state_dict(), beta=model.beta,
                      cost_metric=cost_metric, cost_weight=cost_weight,
                      contradiction_reward=contradiction_reward, history=history, selection=selection, **metadata)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fit(data, *, epochs=20, width=64, rounds=3, batch_size=4, lr=1e-4, seed=0,
        beta=3.0, entropy_weight=.01, cost_weight=1.0,
        contradiction_reward=1.0, verbose=False, output=None, checkpoint_every=10, **limits):
    """Constant-prior REINFORCE; rolling/best snapshots, not optimizer-resume state."""
    _weights(cost_weight, contradiction_reward)
    if min(epochs, batch_size, lr) <= 0 or not isfinite(lr):
        raise ValueError("Epochs, batch size and learning rate must be positive")
    if any(not isfinite(value) or value < 0 for value in (beta, entropy_weight)):
        raise ValueError("Beta and entropy weight must be finite and nonnegative")
    if not isinstance(checkpoint_every, int) or isinstance(checkpoint_every, bool) or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be a positive integer")
    if output is not None:
        output = Path(output)
        best_path = output.with_stem(output.stem + ".best")
        if output.exists() or best_path.exists():
            raise FileExistsError("Output already exists; choose a new filename")
        output.parent.mkdir(parents=True, exist_ok=True)
    state = _state(data)
    torch.manual_seed(seed)
    model = Policy(data["network"], width, rounds, **limits)
    model.reward_colors, model.beta = data["reward_colors"], beta
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng, samples, history, baseline = Random(seed), list(data["samples"]), [], 0.0
    second_moment, best_reward = 1.0, -float("inf")
    for epoch in range(epochs):
        rng.shuffle(samples)
        totals = dict(reward=0.0, gain=0, treewidth=0, cost_s=0.0, selected=0, useful=0, contradictions=0, rejected=0, entropy=0.0)
        for start in range(0, len(samples), batch_size):
            batch, losses, rewards = samples[start:start + batch_size], [], []
            optimizer.zero_grad()
            for sample in batch:
                with _closed(state, sample, model.network["colors"]):
                    log_prob, entropy, metrics = trial(model, state, cost_weight=cost_weight,
                                                       contradiction_reward=contradiction_reward)
                advantage = (metrics["reward"] - baseline) / max(1.0, sqrt(second_moment))
                losses.append(-advantage * log_prob - entropy_weight * entropy)
                rewards.append(metrics["reward"])
                totals["entropy"] += float(entropy.detach())
                for key, value in metrics.items():
                    totals[key] += value
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            second_moment = .9 * second_moment + .1 * sum((reward - baseline) ** 2 for reward in rewards) / len(rewards)
            baseline = .9 * baseline + .1 * sum(rewards) / len(rewards)
        history.append(dict(epoch=epoch + 1, beta=model.beta, states=len(samples), cost_metric=COST_METRIC,
                            **{key: value / len(samples) for key, value in totals.items()}))
        if verbose:
            print(f"epoch {epoch + 1}/{epochs}: beta {model.beta:.3f}, reward {history[-1]['reward']:.4f}, "
                  f"avg treewidth {history[-1]['treewidth']:.2f}, "
                  f"useful {totals['useful']}/{totals['selected']}, skip {len(samples) - totals['selected']}/{len(samples)}, "
                  f"entropy {history[-1]['entropy']:.3f}, rejected {totals['rejected']}", flush=True)
        if output is not None and ((epoch + 1) % checkpoint_every == 0 or epoch + 1 == epochs):
            selection = dict(source="training", beta=beta, **evaluate(
                model, data, cost_weight=cost_weight, contradiction_reward=contradiction_reward, state=state))
            history[-1]["selection_reward"] = selection["reward"]
            if selection["reward"] > best_reward:
                _save(best_path, model, history, cost_weight, contradiction_reward, selection)
                best_reward = selection["reward"]
            _save(output, model, history, cost_weight, contradiction_reward, selection)
            if verbose:
                print(f"Checkpoint epoch {epoch + 1}: greedy training reward {selection['reward']:.4f} "
                      f"at beta {beta:g}; wrote {output}", flush=True)
    return model.eval(), history


@torch.no_grad()
def evaluate(model, data, *, cost_weight=1.0, contradiction_reward=1.0, state=None):
    _weights(cost_weight, contradiction_reward)
    if model.network != data["network"] or model.reward_colors != data["reward_colors"]:
        raise ValueError("Policy and evaluation data must share network schema and reward colors")
    state, totals = _state(data) if state is None else state, {}
    for sample in data["samples"]:
        with _closed(state, sample, model.network["colors"]):
            _, _, metrics = trial(model, state, greedy=True, cost_weight=cost_weight,
                                  contradiction_reward=contradiction_reward)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0) + value
    return dict(states=len(data["samples"]), cost_metric=COST_METRIC,
                **{key: value / len(data["samples"]) for key, value in totals.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Domain-state JSON, e.g. experiments/cluster_selection/data/training.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="Rolling/final checkpoint; also writes a .best sibling")
    mode.add_argument("--checkpoint", type=Path, help="Evaluate a checkpoint on these states")
    defaults = dict(epochs=20, width=64, rounds=3, batch_size=4, seed=0, threads=1,
                    lr=1e-4, beta=3.0, entropy_weight=.01, checkpoint_every=10,
                    cost_weight=1.0, contradiction_reward=1.0, max_cores=10,
                    max_open_colors=4, max_work=1_000_000, max_size=100_000, heuristic_candidates=32)
    for key, default in defaults.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(default), default=default,
                            help={"cost_weight": "Penalty per unit of estimated treewidth",
                                  "beta": "Constant heuristic strength when training (default: 3); checkpoints retain their saved beta",
                                  "heuristic_candidates": "Actual heuristic proposals per state; 0 disables guidance",
                                  "checkpoint_every": "Save/evaluate every N epochs, and always the last"}.get(key))
    args = parser.parse_args()
    if args.output and any(path.exists() for path in (args.output, args.output.with_stem(args.output.stem + ".best"))):
        parser.error("Output already exists; choose a new filename")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    data = json.loads(args.data.read_text())
    if args.checkpoint:
        print(json.dumps(evaluate(load(args.checkpoint), data, cost_weight=args.cost_weight,
                                  contradiction_reward=args.contradiction_reward)))
        print("Closed-state evaluation only; measure full MAC time and nodes separately.")
        return
    options = {key: getattr(args, key) for key in defaults if key != "threads"}
    print(f"Training construction on {len(data['samples'])} states; candidate labels are ignored; "
          f"cost is {COST_METRIC}.", flush=True)
    fit(data, verbose=True, output=args.output, **options)
    print(f"Wrote {args.output} and its .best sibling (training-state selection, not held-out validation).")


if __name__ == "__main__":
    main()
