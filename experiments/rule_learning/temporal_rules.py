"""Time-ordering rules on a temporal event graph, exact against binned.

Task: ordinary filtered link prediction on ICEWS14, with rules whose bodies may carry a comparison
between the timestamps of the two premise events. Three arms share the candidate bodies, the greedy
selection and the evaluator; they differ only in whether the time comparison is available and whether it
is exact:

    A  chain only, no time comparison
    B  chain with the exact comparison t(X,Z) < t(Z,Y)
    C  chain with the comparison on discretised time, k quantile bins, best k

The falsifier for the numerical leg is fixed in advance: if C matches B within the bootstrap interval,
the exact operator buys nothing on real data and the claim dies here.

Usage:
    python temporal_rules.py [--dir data/icews14] [--maxBodies 3000] [--rules 20]
"""

import argparse
import os
import time

import numpy as np
from scipy import sparse

import boost as B
import kg_rules as K

HERE = os.path.dirname(os.path.abspath(__file__))


def load_events(path, entIndex, relIndex, dateIndex):
    """Events as indices, de-duplicated on (head, relation, tail); the earliest time wins."""
    rows, cols, rIdx, times = [], [], [], []
    seen = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            h, r, t, d = parts[0], parts[1], parts[2], parts[3]
            if h not in entIndex or t not in entIndex or r not in relIndex:
                continue
            key = (entIndex[h], relIndex[r], entIndex[t])
            day = dateIndex.get(d)
            if day is None:
                continue
            if key not in seen or day < seen[key]:
                seen[key] = day
    for (h, r, t), d in seen.items():
        rows.append(h)
        cols.append(t)
        rIdx.append(r)
        times.append(d)
    return (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64),
            np.array(rIdx, dtype=np.int64), np.array(times, dtype=np.int64))


def relation_edges(rows, rIdx, cols, times, n, nRel):
    """Per relation, the outgoing and incoming edge lists with their timestamps."""
    out = [[[] for _ in range(n)] for _ in range(nRel)]
    inn = [[[] for _ in range(n)] for _ in range(nRel)]
    for h, r, t, d in zip(rows, rIdx, cols, times):
        out[r][h].append((t, d))
        inn[r][t].append((h, d))
    return out, inn


def chain_matrix(rel, outLists, innLists, n):
    """Plain composition r_a(X,Z) o r_b(Z,Y) as a sparse matrix."""
    rows, cols = [], []
    for z in range(n):
        ins = innLists[rel[0]][z]
        outs = outLists[rel[1]][z]
        if not ins or not outs:
            continue
        for x, _ in ins:
            for y, _ in outs:
                rows.append(x)
                cols.append(y)
    if not rows:
        return sparse.csr_matrix((n, n))
    data = np.ones(len(rows), dtype=np.float64)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def time_matrix(rel, outLists, innLists, n, binning=None):
    """Composition keeping only the witnesses whose two events are ordered in time."""
    rows, cols = [], []
    for z in range(n):
        ins = innLists[rel[0]][z]
        outs = outLists[rel[1]][z]
        if not ins or not outs:
            continue
        yArr = np.array([y for y, _ in outs], dtype=np.int64)
        tOut = np.array([d for _, d in outs], dtype=np.int64)
        if binning is not None:
            tOut = binning[tOut]
        for x, dIn in ins:
            threshold = dIn if binning is None else binning[dIn]
            keep = yArr[tOut > threshold]
            if keep.size:
                rows.extend([x] * keep.size)
                cols.extend(keep.tolist())
    if not rows:
        return sparse.csr_matrix((n, n))
    data = np.ones(len(rows), dtype=np.float64)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def learn(headFacts, mats, nRel, ruleCount, maxBodies, n, minSupport=3):
    """Greedy residual selection over a fixed body pool with per-body matrices."""
    ruleSets = {}
    order = np.argsort([-m.nnz for m in mats])[:maxBodies]
    pool = [int(i) for i in order]
    for r in range(nRel):
        ruleSets[r] = []
        xs = headFacts[r][0]
        if xs.size == 0:
            continue
        ys = headFacts[r][1]
        keys = xs.astype(np.int64) * n + ys
        covered = np.zeros(keys.size, dtype=bool)
        baseRate = min(xs.size / float(n * n), 1.0)
        chosen = []
        for _ in range(ruleCount):
            best = None
            for k in pool:
                if k in chosen:
                    continue
                hits = mats[k][xs, ys].A1 > 0
                support = float(hits.sum())
                if support < minSupport:
                    continue
                residual = int((hits & ~covered).sum())
                if residual < minSupport:
                    continue
                precision = residual / max(float(mats[k].nnz), 1.0)
                gain = B.info_gain(residual, precision, baseRate)
                if gain > 0 and (best is None or gain > best[0]):
                    best = (gain, k, hits, support)
            if best is None:
                break
            gain, k, hits, support = best
            covered |= hits
            chosen.append(k)
            chosenSet = set(chosen)
            ruleSets[r].append({"matrix": mats[k], "confidence": support / max(float(mats[k].nnz), 1.0),
                                "body": ("body", k)})
            del chosenSet
    return ruleSets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=os.path.join(HERE, "data", "icews14"))
    parser.add_argument("--maxBodies", type=int, default=3000)
    parser.add_argument("--rules", type=int, default=20)
    parser.add_argument("--maxCandidates", type=int, default=1200)
    parser.add_argument("--dumpRanks", default=None)
    args = parser.parse_args()

    t0 = time.time()
    train = K.load_triples(os.path.join(args.dir, "train.txt"))
    entities, relations, rows, cols, rIdx = train
    n, nRel = len(entities), len(relations)
    entIndex = {e: i for i, e in enumerate(entities)}
    relIndex = {r: i for i, r in enumerate(relations)}

    dates = set()
    for name in ("train.txt", "valid.txt", "test.txt"):
        with open(os.path.join(args.dir, name), encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    dates.add(parts[3])
    dateIndex = {d: i for i, d in enumerate(sorted(dates))}

    tr = load_events(os.path.join(args.dir, "train.txt"), entIndex, relIndex, dateIndex)
    outLists, innLists = relation_edges(tr[0], tr[2], tr[1], tr[3], n, nRel)
    print("{} entities, {} relations, {} train events, {} dates, loaded in {:.1f}s".format(
        n, nRel, tr[0].size, len(dateIndex), time.time() - t0), flush=True)

    # candidate relation pairs, ranked by two-hop support, capped for the census
    jointSupport = []
    for a in range(nRel):
        for b in range(nRel):
            count = 0
            for z in range(n):
                count += len(innLists[a][z]) * len(outLists[b][z])
                if count > 400000:
                    break
            if count:
                jointSupport.append((count, a, b))
    jointSupport.sort(reverse=True)
    pairs = [(a, b) for _, a, b in jointSupport[: args.maxCandidates]]
    print("{} relation pairs with two-hop support, using {}".format(len(jointSupport), len(pairs)),
          flush=True)

    headFacts = {}
    for r in range(nRel):
        own = np.nonzero(tr[2] == r)[0]
        headFacts[r] = (tr[0][own], tr[1][own])

    t1 = time.time()
    plain = [chain_matrix(p, outLists, innLists, n) for p in pairs]
    exact = [time_matrix(p, outLists, innLists, n) for p in pairs]
    print("priced {} bodies and {} with the exact time comparison in {:.1f}s".format(
        len(pairs), len(pairs), time.time() - t1), flush=True)

    bins = {}
    for k in (5, 10, 20, 50):
        quantiles = np.quantile(np.arange(len(dateIndex)), np.linspace(0, 1, k + 1)[1:-1])
        bins[k] = np.digitize(np.arange(len(dateIndex)), quantiles)

    trainIdx = list(zip(tr[0].tolist(), tr[2].tolist(), tr[1].tolist()))
    validPath = os.path.join(args.dir, "valid.txt")
    testPath = os.path.join(args.dir, "test.txt")

    def run(name, mats, extra=None):
        base = learn(headFacts, mats, nRel, args.rules, args.maxBodies, n)
        sets = {}
        for r, rules in base.items():
            sets[r] = list(rules)
        if extra is not None:
            extraSets = learn(headFacts, extra, nRel, args.rules, args.maxBodies, n)
            for r, rules in extraSets.items():
                sets[r].extend(rules)
            for r in sets:
                seen, kept = set(), []
                for rule in sets[r]:
                    if id(rule["matrix"]) in seen:
                        continue
                    seen.add(id(rule["matrix"]))
                    kept.append(rule)
                sets[r] = kept[: args.rules]
        dump = None if args.dumpRanks is None else "{}.{}.npz".format(args.dumpRanks,
                                                                    name.split()[0])
        mrr = B.evaluate({}, sets, n, relations, entIndex, relIndex, trainIdx, [validPath], testPath,
                         aggregation="sum", direction="both", dumpPath=dump)
        fired = sum(len(v) for v in sets.values())
        print("{:28s} both MRR {:.4f}  rules {}".format(name, mrr, fired), flush=True)
        return mrr

    run("A chain, no time", plain)
    run("B chain + exact time", exact)
    for k in (5, 10, 50):
        binned = [time_matrix(p, outLists, innLists, n, binning=bins[k]) for p in pairs]
        run("C chain + {} bins".format(k), binned)


if __name__ == "__main__":
    main()
