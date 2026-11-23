"""Can the learner recover a rule that contains a numerical comparison?

The graph is planted, so the answer is known: h(x,y) holds when some z has r(x,z) and s(z,y) and the
feature of x exceeds the feature of z. A companion relation d(x,y) is the same chain without the gate,
so an arm that cannot express comparisons still has real structure to learn and is not being compared
against nothing.

Both arms are the same learner on the same split. The only difference is whether comparison bodies are
available.

Usage:
    python numeric_rule.py [--entities 800] [--bodies 2000]
"""

import argparse
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def build(nEnt, seed=0, k=3):
    rng = np.random.default_rng(seed)
    ages = rng.integers(0, 80, size=nEnt).astype(np.float64)
    R = np.zeros((nEnt, nEnt), dtype=bool)
    S = np.zeros((nEnt, nEnt), dtype=bool)
    R[np.arange(nEnt)[:, None], rng.integers(0, nEnt, size=(nEnt, k))] = True
    S[np.arange(nEnt)[:, None], rng.integers(0, nEnt, size=(nEnt, k))] = True
    chain = (R.astype(np.int32) @ S.astype(np.int32)) > 0
    # the gate: the head entity is older than the intermediate one
    gate = np.zeros((nEnt, nEnt), dtype=bool)
    rx, rz = np.nonzero(R)
    gate[rx, rz] = ages[rx] > ages[rz]
    gated = (gate.astype(np.int32) @ S.astype(np.int32)) > 0
    return ages, {"r": R, "s": S, "d": chain, "h": gated}


def write(path, facts, ages, seed=0):
    rng = np.random.default_rng(seed)
    names = ["e{}".format(i) for i in range(ages.size)]
    train, test = [], []
    for rel, mat in facts.items():
        xs, ys = np.nonzero(mat)
        take = rng.random(xs.size) < 0.2
        for x, y, isTest in zip(xs, ys, take):
            line = "{}\t{}\t{}\n".format(names[x], rel, names[y])
            (test if isTest else train).append(line)
    os.makedirs(path, exist_ok=True)
    for name, part in (("train", train), ("valid", train[: len(train) // 10]), ("test", test)):
        with open(os.path.join(path, name + ".txt"), "w") as f:
            f.writelines(part)
    with open(os.path.join(path, "features.txt"), "w") as f:
        f.writelines("{}\t{:.1f}\n".format(names[i], ages[i]) for i in range(ages.size))
    return len(train), len(test)


def run(path, features):
    cmd = [PY, os.path.join(HERE, "boost.py"), "--dir", path, "--depth", "2",
           "--maxBodies", "2000", "--inverse", "--aggregation", "sum", "--topology", "chain"]
    if features:
        cmd += ["--features", os.path.join(path, "features.txt")]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stdout[-2000:] + out.stderr[-2000:])
    mrr = re.search(r"^both  MRR ([0-9.]+)", out.stdout, re.M)
    cmpRules = [line.strip() for line in out.stdout.splitlines() if "feature(X)" in line]
    return float(mrr.group(1)), cmpRules, out.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=int, default=800)
    parser.add_argument("--bodies", type=int, default=2000)
    args = parser.parse_args()

    ages, facts = build(args.entities)
    path = os.path.join(HERE, "data", "numeric_planted")
    nTrain, nTest = write(path, facts, ages)
    print("{} entities, {} train and {} test triples".format(ages.size, nTrain, nTest), flush=True)

    plain, plainRules, _ = run(path, features=False)
    gated, gatedRules, output = run(path, features=True)

    print("\narm                      both MRR   comparison rules selected")
    print("chain only               {:.4f}     {}".format(plain, len(plainRules)))
    print("chain + comparisons      {:.4f}     {}".format(gated, len(gatedRules)))
    print("\ndelta {:+.4f}".format(gated - plain))
    for line in gatedRules[:3]:
        print("  " + line)


if __name__ == "__main__":
    main()
