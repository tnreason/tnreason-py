"""The selection tensor as something to differentiate, not something to search.

The discrete learner picks rules greedily. Here the same candidate bodies are a design matrix and the
rule set is a weight vector: score(x, y) for a head relation is a non-negative combination of the
bodies that fire on the pair. That vector is fit by minimising a logistic loss, so the selection tensor
F[X, L] is a parameter rather than the output of a search.

The comparison is deliberately narrow. Both arms see the same bodies, keep the same number of rules per
head, and are scored by the same evaluator; the only difference is how the rule weights are obtained.
That makes it a test of the objective, not of the hypothesis space.

Usage:
    python train_soft.py [--dir data/kinship] [--rules 20]
"""

import argparse
import os
import time

import numpy as np
from scipy import sparse

import boost as B
import kg_rules as K


def build(dirPath, inverse=True):
    entities, relations, rows, cols, rIdx = K.load_triples(os.path.join(dirPath, "train.txt"))
    n, nRel = len(entities), len(relations)
    headCores = K.relation_cores(rows, cols, rIdx, n, nRel)
    cores = headCores + [c.T.tocsr() for c in headCores] if inverse else headCores
    nAtom = len(cores)
    dom, rang = B.domains_ranges(cores)
    joint = np.array([B.overlaps(rang[a], dom) for a in range(nAtom)])
    bodies = [(a, b) for a in range(nAtom) for b in range(nAtom) if joint[a, b]]
    return entities, relations, rows, cols, rIdx, cores, bodies


def binarised(bodies, cores):
    return [(B.chain(cores, body) > 0).astype(np.float64).tocsr() for body in bodies]


def design(mats, xs, ys):
    """Which bodies fire on each candidate pair, as a dense 0/1 matrix."""
    out = np.zeros((xs.size, len(mats)), dtype=np.float64)
    for l, m in enumerate(mats):
        out[:, l] = np.asarray(m[xs, ys]).ravel()
    return out


def listwise_train(A, groups, labels, iters=300, lr=0.5, l2=1e-3):
    """Softmax over the candidates of each query, which is what the ranking metric measures.

    Per-triple logistic treats every corrupted tail as an independent decision and lets an easy negative
    dominate the gradient; a query's candidates are only ever compared with each other, so the loss has
    to be taken over that set.
    """
    nGroups = int(groups.max()) + 1
    w = np.zeros(A.shape[1])
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    for t in range(1, iters + 1):
        z = A @ w
        top = np.full(nGroups, -np.inf)
        np.maximum.at(top, groups, z)
        e = np.exp(z - top[groups])
        denom = np.bincount(groups, weights=e, minlength=nGroups)
        p = e / np.maximum(denom[groups], 1e-300)
        grad = A.T @ (p - labels) / nGroups + l2 * w
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad * grad
        w -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        np.maximum(w, 0.0, out=w)
    return w


def logit_train(A, labels, iters=300, lr=0.5, l2=1e-3):
    """Non-negative logistic regression by Adam; returns the weight vector."""
    w = np.zeros(A.shape[1])
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    for t in range(1, iters + 1):
        z = A @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = A.T @ (p - labels) / A.shape[0] + l2 * w
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad * grad
        w -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        np.maximum(w, 0.0, out=w)
    return w


def rule_sets(bodies, mats, weights, ruleCount, coverage=None):
    sets = {}
    for r, bodyWeights in enumerate(weights):
        rankBy = bodyWeights if coverage is None else bodyWeights * coverage[r]
        order = np.argsort(-rankBy)[:ruleCount]
        order = [int(l) for l in order if bodyWeights[l] > 1e-9]
        sets[r] = [{"body": bodies[l], "confidence": float(bodyWeights[l]), "matrix": mats[l]}
                   for l in order]
    return sets


def confidence_sets(bodies, mats, rows, cols, rIdx, nRel, n, ruleCount, minSupport=2):
    """The discrete baseline on the same bodies: precision of each body on the training facts."""
    byRel = {}
    for x, r, y in zip(rows, cols, rIdx):
        byRel.setdefault(r, []).append((x, y))
    sets = {}
    for r in range(nRel):
        xs = np.array([x for x, _ in byRel.get(r, [])], dtype=np.int64)
        ys = np.array([y for _, y in byRel.get(r, [])], dtype=np.int64)
        if xs.size == 0:
            sets[r] = []
            continue
        hits = np.array([np.asarray(m[xs, ys]).sum() for m in mats])
        totals = np.array([m.nnz for m in mats], dtype=np.float64)
        confidence = np.where(totals > 0, hits / np.maximum(totals, 1.0), 0.0)
        order = np.argsort(-confidence)
        chosen = [int(l) for l in order if hits[l] >= minSupport][:ruleCount]
        sets[r] = [{"body": bodies[l], "confidence": float(confidence[l]), "matrix": mats[l]}
                   for l in chosen]
    return sets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "data", "kinship"))
    parser.add_argument("--rules", type=int, default=20)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--positives", type=int, default=400)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--rank", default="weight", choices=["weight", "contribution"])
    parser.add_argument("--dumpRanks", default=None, help="write tail and head ranks to this npz path")
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--iters", type=int, default=300)
    args = parser.parse_args()

    t0 = time.time()
    entities, relations, rows, cols, rIdx, cores, bodies = build(args.dir)
    n, nRel = len(entities), len(relations)
    mats = binarised(bodies, cores)
    print("{} entities, {} relations, {} atoms, {} bodies priced in {:.1f}s".format(
        n, nRel, len(cores), len(bodies), time.time() - t0), flush=True)

    entIndex = {e: i for i, e in enumerate(entities)}
    relIndex = {r: i for i, r in enumerate(relations)}
    trainIdx = list(zip(rows.tolist(), rIdx.tolist(), cols.tolist()))
    facts = set(trainIdx)

    rng = np.random.default_rng(0)
    weights = []
    t1 = time.time()
    for r in range(nRel):
        own = np.nonzero(rIdx == r)[0]
        if own.size == 0:
            weights.append(np.zeros(len(bodies)))
            continue
        queries = rng.permutation(np.unique(rows[own]))[: args.queries]
        groupX, groupY, groupId, labels = [], [], [], []
        for g, x in enumerate(queries):
            tails = cols[own][rows[own] == x]
            groupX.append(int(x))
            groupY.append(int(tails[0]))
            groupId.append(g)
            labels.append(1.0)
            for _ in range(args.negatives):
                yc = int(rng.integers(0, n))
                if (int(x), int(r), yc) in facts:
                    continue
                groupX.append(int(x))
                groupY.append(yc)
                groupId.append(g)
                labels.append(0.0)
        A = design(mats, np.array(groupX), np.array(groupY))
        weights.append(listwise_train(A, np.array(groupId), np.array(labels),
                                      iters=args.iters, l2=args.l2))
    print("fitted {} heads by logistic loss in {:.1f}s".format(nRel, time.time() - t1), flush=True)

    # coverage of each body on the training facts of each head, for ranking by contribution
    coverage = np.zeros((nRel, len(bodies)))
    for r in range(nRel):
        own = np.nonzero(rIdx == r)[0]
        if own.size == 0:
            continue
        xs = rows[own].astype(np.int64)
        ys = cols[own].astype(np.int64)
        coverage[r] = [float(np.asarray(m[xs, ys]).sum()) for m in mats]
    soft = rule_sets(bodies, mats, weights, args.rules,
                     coverage if args.rank == "contribution" else None)
    discrete = confidence_sets(bodies, mats, rows, cols, rIdx, nRel, n, args.rules)

    validPath = os.path.join(args.dir, "valid.txt")
    testPath = os.path.join(args.dir, "test.txt")
    for name, sets in (("confidence", discrete), ("logistic", soft)):
        dump = None if args.dumpRanks is None else "{}.{}.npz".format(args.dumpRanks, name)
        mrr = B.evaluate(cores, sets, n, relations, entIndex, relIndex, trainIdx,
                         [validPath], testPath, aggregation="sum", direction="both", dumpPath=dump)
        fired = sum(len(v) for v in sets.values())
        print("{:12s} both MRR {:.4f}  rules {}".format(name, mrr, fired), flush=True)


if __name__ == "__main__":
    main()
