"""Supervised cluster prediction; evaluation times are rollback-inclusive proxies."""
import argparse
import json
from math import log, log1p
from pathlib import Path
from random import Random
import torch
from torch.nn import functional as F
from .model import Predictor, load


def loss(prediction, candidates):
    target = prediction.new_tensor([[log1p(candidate["gain"]), log(max(candidate["time"], 1e-6)),
                                    float(candidate["contradiction"])] for candidate in candidates])
    return (F.smooth_l1_loss(prediction[:, 0], target[:, 0])
            + 0.1 * F.smooth_l1_loss(prediction[:, 1], target[:, 1])
            + F.binary_cross_entropy_with_logits(prediction[:, 2], target[:, 2]))


def evaluate(model, data, *, cost_weight=1.0, contradiction_reward=1.0):
    """Offline action quality on labeled states, not end-to-end solving time."""
    if data.get("version") != 1 or data["network"] != model.network or not data["samples"]:
        raise ValueError("Expected nonempty test data with the checkpoint's exact network schema")
    if set(data["reward_colors"]) != set(model.reward_colors or ()):
        raise ValueError("Test data and checkpoint must use the same reward colors")
    selected = useful = 0
    reward = regret = 0.0
    for sample in data["samples"]:
        candidates = sample["candidates"]
        values = [candidate["gain"] + contradiction_reward * candidate["contradiction"]
                  - cost_weight * candidate["time"] for candidate in candidates]
        idx = model.select(sample["domains"], candidates, cost_weight=cost_weight,
                           contradiction_reward=contradiction_reward)
        value = 0.0 if idx is None else values[idx]
        if idx is not None:
            selected += 1
            useful += candidates[idx]["gain"] > 0 or candidates[idx]["contradiction"]
        reward += value
        regret += max([0.0, *values]) - value
    count = len(data["samples"])
    return dict(states=count, selected=selected, useful=useful,
                mean_reward=reward / count, mean_regret=regret / count)


def fit(data, *, epochs=20, width=64, rounds=3, batch_size=4, lr=1e-3, seed=0, verbose=False):
    """One state encoding per candidate group; average state losses within batches."""
    if data.get("version") != 1 or not data["samples"] or any(not sample["candidates"] for sample in data["samples"]):
        raise ValueError("Expected a version-1 dataset with nonempty candidate groups")
    if min(epochs, batch_size, lr) <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive")
    torch.manual_seed(seed)
    model = Predictor(data["network"], width=width, rounds=rounds)
    model.reward_colors = data["reward_colors"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng, samples, history = Random(seed), list(data["samples"]), []
    for epoch in range(epochs):
        rng.shuffle(samples)
        total = 0.0
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            optimizer.zero_grad()
            value = sum(loss(model(sample["domains"], sample["candidates"]), sample["candidates"])
                        for sample in batch) / len(batch)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(value.detach()) * len(batch)
        history.append(total / len(samples))
        if verbose:
            print(f"epoch {epoch + 1}/{epochs}: loss {history[-1]:.5f}", flush=True)
    return model.eval(), history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, help="New checkpoint filename when training")
    parser.add_argument("--checkpoint", type=Path, help="Test an existing checkpoint instead of training")
    parser.add_argument("--cost-weight", type=float, default=1.0)
    parser.add_argument("--contradiction-reward", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if bool(args.output) == bool(args.checkpoint):
        parser.error("Choose --output for training or --checkpoint for testing")
    if args.output and args.output.exists():
        parser.error("Output already exists; choose a new filename")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    data = json.loads(args.data.read_text())
    if args.checkpoint:
        result = evaluate(load(args.checkpoint), data, cost_weight=args.cost_weight,
                          contradiction_reward=args.contradiction_reward)
        print(json.dumps(result))
        print("Offline labeled-state evaluation only; candidate-list oracle regret is lower-is-better.")
        return
    print(f"Training on {len(data['samples'])} domain states...", flush=True)
    model, history = fit(data, epochs=args.epochs, width=args.width, rounds=args.rounds,
                         batch_size=args.batch_size, lr=args.lr, seed=args.seed, verbose=True)
    with args.output.open("xb") as stream:
        torch.save(dict(network=model.network, config=model.config, weights=model.state_dict(),
                        reward_colors=model.reward_colors, version=1, history=history), stream)
    print(f"Wrote {args.output}; this is training loss, not fresh-puzzle solver performance.")


if __name__ == "__main__":
    main()
