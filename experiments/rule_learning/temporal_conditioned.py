"""Time-conditioned evaluation: does a temporal atom help when the query has a time?

The previous run ranked candidates on a static snapshot, so a body whose witnesses had to be earlier could
only lose support. Here the query carries its timestamp and only facts up to that time are eligible,
which is the protocol under which an ordering atom can do work.

Arms, identical rules and identical eligibility (facts at or before the query time):

    A  no ordering atom
    B  exact atom: the second premise is strictly later than the first
    C  binned atom: the same on quantile bins of the timestamp

Rules are relation pairs scored by their precision on the training period, which is the cheapest learner
that keeps the comparison about the atom rather than about selection.

Usage:
    python temporal_conditioned.py [--dir data/icews14] [--rules 200] [--maxTest 2000]
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np

import kg_rules as K

HERE = os.path.dirname(os.path.abspath(__file__))


def load_all(dirPath):
    events, dates = [], set()
    for name in ("train.txt", "valid.txt", "test.txt"):
        with open(os.path.join(dirPath, name), encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                events.append((parts[0], parts[1], parts[2], parts[3], name))
                dates.add(parts[3])
    return events, sorted(dates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=os.path.join(HERE, "data", "icews14"))
    parser.add_argument("--rules", type=int, default=200)
    parser.add_argument("--maxTest", type=int, default=2000)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--modes", default="none,exact,binned,distinct,reverse,random")
    args = parser.parse_args()

    t0 = time.time()
    events, dates = load_all(args.dir)
    entities = sorted({e for h, _, t, _, _ in events for e in (h, t)})
    relations = sorted({r for _, r, _, _, _ in events})
    entIndex = {e: i for i, e in enumerate(entities)}
    relIndex = {r: i for i, r in enumerate(relations)}
    dateIndex = {d: i for i, d in enumerate(dates)}
    n, nRel = len(entities), len(relations)

    out = defaultdict(list)
    known = defaultdict(set)
    train = []
    for h, r, t, d, part in events:
        hi, ti, ri, di = entIndex[h], entIndex[t], relIndex[r], dateIndex[d]
        out[hi].append((ri, ti, di))
        known[(hi, ri)].add(ti)
        if part == "train.txt":
            train.append((hi, ri, ti, di))
    print("{} events, {} entities, {} relations, {} dates, prepared in {:.1f}s".format(
        len(events), n, nRel, len(dates), time.time() - t0), flush=True)

    # rules: relation pairs scored by precision of the second premise given the first, on the train period
    firstCount = defaultdict(int)
    pairCount = defaultdict(int)
    byRel = defaultdict(list)
    for hi, ri, ti, di in train:
        for r2, t2, d2 in out[ti]:
            pairCount[(ri, r2)] += 1
        firstCount[ri] += 1
        byRel[ri].append((ti, di))
    ranked = sorted(((count / max(firstCount[ri], 1), count, ri, r2)
                     for (ri, r2), count in pairCount.items()), reverse=True)
    rules = [(ri, r2, weight) for weight, count, ri, r2 in ranked[: args.rules]
             if count >= 5]
    print("{} rules selected from {} relation pairs".format(len(rules), len(pairCount)), flush=True)
    rngW = np.random.default_rng(0)
    gapDays = args.gap
    rulesByFirst = defaultdict(list)
    for ri, r2, weight in rules:
        rulesByFirst[ri].append((r2, weight))

    quantiles = np.quantile(np.arange(len(dates)), np.linspace(0, 1, args.bins + 1)[1:-1])
    binOf = np.digitize(np.arange(len(dates)), quantiles)

    test = [(entIndex[h], relIndex[r], entIndex[t], dateIndex[d])
            for h, r, t, d, part in events if part == "test.txt"][: args.maxTest]
    # tie-break prior: how often an entity appears as a tail of this relation in the training period
    priorCount = defaultdict(lambda: np.zeros(n, dtype=np.float64))
    for hi, ri, ti, di in train:
        priorCount[ri][ti] += 1.0
    for ri in priorCount:
        priorCount[ri] /= max(priorCount[ri].max(), 1.0)
    print("{} test queries".format(len(test)), flush=True)

    def scoreQuery(hi, ri, tq, mode):
        scores = np.zeros(n, dtype=np.float64)
        for r1, ti, d1 in out[hi]:
            if d1 > tq:
                continue
            for r2, weight in rulesByFirst.get(r1, ()):
                for r3, yi, d2 in out[ti]:
                    if r3 != r2 or d2 > tq:
                        continue
                    if mode == "exact" and not d1 < d2:
                        continue
                    if mode.startswith("gap") and not (d2 - d1) >= gapDays:
                        continue
                    if mode == "binned" and not binOf[d1] < binOf[d2]:
                        continue
                    if mode == "distinct" and binOf[d1] == binOf[d2]:
                        continue
                    if mode == "reverse" and not binOf[d1] > binOf[d2]:
                        continue
                    if mode.startswith("random") and not rngW.random() < 0.5:
                        continue
                    if mode.startswith("gap") and not (d2 - d1) >= gapDays:
                        continue
                    scores[yi] += weight
        return scores

    results = {}
    for mode in args.modes.split(","):
        ranks = []
        t1 = time.time()
        for hi, ri, ti, tq in test:
            scores = scoreQuery(hi, ri, tq, mode)
            # a score of zero means no rule fired, which is not evidence of a wrong answer: break ties
            # by the relation's empirical tail marginal, then by entity index
            scores = scores + 1e-6 * priorCount[ri] - 1e-9 * np.arange(n, dtype=np.float64)
            blocked = known[(hi, ri)] - {ti}
            for b in blocked:
                scores[b] = -np.inf
            target = scores[ti]
            ranks.append(1 + int((scores > target).sum()))
        ranks = np.array(ranks, dtype=np.float64)
        results[mode] = ranks
        print("{:8s} MRR {:.4f}  H@1 {:.4f}  H@10 {:.4f}  ({:.1f}s)".format(
            mode, float((1.0 / ranks).mean()), float((ranks <= 1).mean()),
            float((ranks <= 10).mean()), time.time() - t1), flush=True)

    for a, b in (("none", "exact"), ("binned", "exact"), ("distinct", "binned")):
        diff = float((1.0 / results[b]).mean() - (1.0 / results[a]).mean())
        boots = []
        rng = np.random.default_rng(0)
        for _ in range(2000):
            idx = rng.integers(0, len(results[a]), len(results[a]))
            boots.append(float((1.0 / results[b][idx]).mean() - (1.0 / results[a][idx]).mean()))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print("{} minus {}: {:+.4f}  [{:+.4f}, {:+.4f}]".format(b, a, diff, lo, hi), flush=True)


if __name__ == "__main__":
    main()
