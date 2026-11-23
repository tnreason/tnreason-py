"""Planted gated-composition control: is the tree gain structural or a dataset quirk?

The generating rule is a tree by construction, h(x,y) holds when some z has a(x,z) and b(z,y)
and x has any c(x,w). No chain over the alphabet can express that, which the tree self-check
verifies at small scale. Here the same construction is repeated at knowledge-graph scale with a
train/valid/test split, ten seeds, and the real learner in both arms, so that a null result cannot
be blamed on the graph's density.

Two relations are added so the arms are not starved of ordinary structure: a plain chain
d(x,y) holds when some z has a(x,z) and b(z,y), and a decoy e that is random noise.

Usage:
    python planted_tree.py [--seeds 10] [--entities 1200] [--bodies 2000]
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def build(seed, nEnt, k):
    rng = np.random.default_rng(seed)
    A = np.zeros((nEnt, nEnt), dtype=bool)
    B = np.zeros((nEnt, nEnt), dtype=bool)
    C = np.zeros((nEnt, nEnt), dtype=bool)
    E = np.zeros((nEnt, nEnt), dtype=bool)
    dest = rng.integers(0, nEnt, size=(nEnt, k))
    A[np.arange(nEnt)[:, None], dest] = True
    B[np.arange(nEnt)[:, None], rng.integers(0, nEnt, size=(nEnt, k))] = True
    gateRows = np.where(np.arange(nEnt) % 4 == 0)[0]          # only a quarter of x has a c edge
    C[gateRows[:, None], rng.integers(0, nEnt, size=(gateRows.size, k))] = True
    E[np.random.default_rng(seed + 99).random((nEnt, nEnt)) < 0.01] = True

    A32 = A.astype(np.int32)
    B32 = B.astype(np.int32)
    chainFacts = (A32 @ B32) > 0                              # d: ordinary chain
    treeFacts = chainFacts & C.any(axis=1)[:, None]           # h: chain gated by c(x,.)
    return {"a": A, "b": B, "c": C, "e": E, "d": chainFacts, "h": treeFacts}


def write_split(path, facts, names, rng, testFraction=0.2):
    """Split every relation's facts, writing head/relation/tail triples as tab-separated text."""
    train, test = [], []
    for rel, mat in facts.items():
        xs, ys = np.nonzero(mat)
        take = rng.random(xs.size) < testFraction
        for x, y, isTest in zip(xs, ys, take):
            line = "{}\t{}\t{}\n".format(names[x], rel, names[y])
            (test if isTest else train).append(line)
    with open(os.path.join(path, "train.txt"), "w") as f:
        f.writelines(train)
    with open(os.path.join(path, "test.txt"), "w") as f:
        f.writelines(test)
    with open(os.path.join(path, "valid.txt"), "w") as f:
        f.writelines(train[: len(train) // 10])
    return len(train), len(test)


def run_arm(path, topology, bodies, depth, rules, minSupport):
    cmd = [PY, os.path.join(HERE, "boost.py"), "--dir", path, "--depth", str(depth),
           "--maxBodies", str(bodies), "--rules", str(rules), "--minSupport", str(minSupport),
           "--inverse", "--aggregation", "sum", "--topology", topology]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("arm {} failed:\n{}".format(topology, out.stdout[-2000:] + out.stderr[-2000:]))
    match = re.search(r"^both  MRR ([0-9.]+)", out.stdout, re.M)
    if not match:
        raise RuntimeError("no MRR in arm {} output:\n{}".format(topology, out.stdout[-1500:]))
    learned = re.search(r"learned (\d+) rules", out.stdout)
    return float(match.group(1)), int(learned.group(1)) if learned else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--entities", type=int, default=1200)
    parser.add_argument("--bodies", type=int, default=2000)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--rules", type=int, default=20)
    parser.add_argument("--minSupport", type=int, default=5)
    args = parser.parse_args()

    rows = []
    with tempfile.TemporaryDirectory() as root:
        for seed in range(args.seeds):
            path = os.path.join(root, "seed{}".format(seed))
            os.makedirs(path)
            facts = build(seed, args.entities, 3)
            names = ["e{}".format(i) for i in range(args.entities)]
            nTrain, nTest = write_split(path, facts, names, np.random.default_rng(seed + 1))
            chain, chainRules = run_arm(path, "chain", args.bodies, args.depth, args.rules, args.minSupport)
            tree, treeRules = run_arm(path, "tree", args.bodies, args.depth, args.rules, args.minSupport)
            rows.append((seed, chain, tree, tree - chain, chainRules, treeRules))
            print("seed {}  chain {:.4f} ({} rules)  tree {:.4f} ({} rules)  delta {:+.4f}".format(
                seed, chain, chainRules, tree, treeRules, tree - chain), flush=True)

    deltas = np.array([r[3] for r in rows])
    wins = int((deltas > 0).sum())
    print("\n{} seeds  chain mean {:.4f}  tree mean {:.4f}  delta mean {:+.4f} sd {:.4f}  tree wins {}".format(
        len(rows), np.mean([r[1] for r in rows]), np.mean([r[2] for r in rows]),
        deltas.mean(), deltas.std(ddof=1), wins))
    # sign test: under the null of no effect each seed is a fair coin
    from math import comb
    tail = sum(comb(len(rows), k) for k in range(wins, len(rows) + 1)) / 2 ** len(rows)
    print("sign test p {:.4f} (one-sided, tree strictly better)".format(tail))


if __name__ == "__main__":
    main()
