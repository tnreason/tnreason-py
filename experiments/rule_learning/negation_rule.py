"""Can the learner recover a rule that contains a negated atom?

Planted graph: h(x,y) holds when some z has r(x,z) and s(z,y) and the pair (x,y) is not blocked by b,
where b is a dense relation so that the negation actually removes candidates. An arm without negation is
the same learner on the same split.

Usage:
    python negation_rule.py [--entities 400]
"""

import argparse
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def build(nEnt, seed=0, k=3, density=0.4):
    rng = np.random.default_rng(seed)
    R = np.zeros((nEnt, nEnt), dtype=bool)
    S = np.zeros((nEnt, nEnt), dtype=bool)
    R[np.arange(nEnt)[:, None], rng.integers(0, nEnt, size=(nEnt, k))] = True
    S[np.arange(nEnt)[:, None], rng.integers(0, nEnt, size=(nEnt, k))] = True
    chain = (R.astype(np.int32) @ S.astype(np.int32)) > 0
    # the blocker rules out a share of the chain pairs, not of all entity pairs
    blockers = chain & (rng.random((nEnt, nEnt)) < density)
    return {"r": R, "s": S, "b": blockers, "h": chain & ~blockers}, chain.sum()


def write(path, facts, nEnt, seed=0):
    rng = np.random.default_rng(seed)
    names = ["e{}".format(i) for i in range(nEnt)]
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
    return len(train), len(test)


def run(path, negation, maxBodies):
    cmd = [PY, os.path.join(HERE, "boost.py"), "--dir", path, "--depth", "2",
           "--maxBodies", str(maxBodies), "--inverse", "--aggregation", "sum", "--topology", "chain"]
    if negation is not None:
        cmd += ["--negation", str(negation)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stdout[-2000:] + out.stderr[-2000:])
    mrr = re.search(r"^both  MRR ([0-9.]+)", out.stdout, re.M)
    negRules = [line.strip() for line in out.stdout.splitlines() if "not " in line and "body=" in line]
    return float(mrr.group(1)), negRules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=int, default=400)
    args = parser.parse_args()

    facts, chainCount = build(args.entities)
    path = os.path.join(HERE, "data", "negation_planted")
    nTrain, nTest = write(path, facts, args.entities)
    print("{} entities, {} train and {} test triples, {} gated facts of {} chain pairs".format(
        args.entities, nTrain, nTest, facts["h"].sum(), chainCount), flush=True)

    plain, _ = run(path, None, 1500)
    gated, negRules = run(path, 0.005, 1500)

    print("\narm               both MRR   negation rules selected")
    print("chain only        {:.4f}     {}".format(plain, 0))
    print("chain + negation  {:.4f}     {}".format(gated, len(negRules)))
    print("\ndelta {:+.4f}".format(gated - plain))
    for line in negRules[:3]:
        print("  " + line)


if __name__ == "__main__":
    main()
