"""Paired bootstrap over test triples for two rank dumps from boost.py --dumpRanks.

Each dump holds the tail ranks and the head ranks in evaluation order, which is deterministic
(test triples grouped by head relation, relations in sorted order). Two dumps therefore align
row for row, so the difference can be resampled per triple rather than per direction.

Usage:
    python bootstrap_ci.py a.npz b.npz [--samples 10000] [--label "b minus a"]
"""

import argparse

import numpy as np


def mrr(ranks):
    return float((1.0 / ranks).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--label", default="b minus a")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    da, db = np.load(args.a), np.load(args.b)
    for key in ("tail", "head"):
        if da[key].size != db[key].size:
            raise SystemExit("dump sizes differ for {}: {} vs {}".format(key, da[key].size, db[key].size))

    tailA, tailB = da["tail"], db["tail"]
    headA, headB = da["head"], db["head"]
    if tailA.size != headA.size:
        raise SystemExit("this script expects both directions for every triple")

    contribA = 0.5 * (1.0 / tailA + 1.0 / headA)
    contribB = 0.5 * (1.0 / tailB + 1.0 / headB)
    delta = contribB - contribA
    point = float(delta.mean())

    rng = np.random.default_rng(args.seed)
    n = delta.size
    draws = np.empty(args.samples)
    for i in range(args.samples):
        idx = rng.integers(0, n, n)
        draws[i] = delta[idx].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])

    print("triples {}  tail MRR a {:.4f} b {:.4f}   head MRR a {:.4f} b {:.4f}".format(
        n, mrr(tailA), mrr(tailB), mrr(headA), mrr(headB)))
    print("both-direction MRR a {:.4f}  b {:.4f}".format(contribA.mean(), contribB.mean()))
    print("{}  point {:.4f}  95% CI [{:.4f}, {:.4f}]  {}".format(
        args.label, point, lo, hi,
        "excludes zero" if (lo > 0 or hi < 0) else "INCLUDES ZERO"))


if __name__ == "__main__":
    main()
