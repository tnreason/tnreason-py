"""Step 0: chain-rule scoring on a KG by contraction of stacked relation cores.

A rule  h(Y0,Y2) <- b1(Y0,Y1) ^ b2(Y1,Y2)  is scored by existence semantics:

    support[a,b] = |{ (x,y) in W : (B_a B_b)[x,y] > 0 }|
    bodyNnz[a,b] = nnz( B_a B_b )
    confidence   = support / bodyNnz

B_r = adjacency of relation r, W = observed pairs of the head relation. The
product B_a B_b is the contraction of the stacked core Rel[L, O1, O2] with the
selection mode L fixed to (a,b); the whole score table is one pass over the
selection space, no per-candidate search.

Usage:
    python kg_rules.py --selftest
    python kg_rules.py --triples data/wn18rr/train.txt --topk 5 --gate
"""

import argparse
import os
import time

import numpy as np
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))


def load_triples(path):
    """head <tab> relation <tab> tail -> entity list, relation list, index arrays."""
    heads, rels, tails = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            heads.append(parts[0])
            rels.append(parts[1])
            tails.append(parts[2])
    entities = sorted(set(heads) | set(tails))
    relations = sorted(set(rels))
    entIndex = {e: i for i, e in enumerate(entities)}
    relIndex = {r: i for i, r in enumerate(relations)}
    rows = np.array([entIndex[h] for h in heads], dtype=np.int32)
    cols = np.array([entIndex[t] for t in tails], dtype=np.int32)
    rIdx = np.array([relIndex[r] for r in rels], dtype=np.int32)
    return entities, relations, rows, cols, rIdx


def relation_cores(rows, cols, rIdx, nEnt, nRel):
    """One sparse adjacency per relation: the slices of Rel[L, O1, O2]."""
    cores = []
    for a in range(nRel):
        sel = rIdx == a
        cores.append(sparse.csr_matrix((np.ones(int(sel.sum()), dtype=np.float32), (rows[sel], cols[sel])),
                                      shape=(nEnt, nEnt)))
    return cores


def score_chains(cores, head, minSupport=1):
    """support[a,b], bodyNnz[a,b], confidence[a,b] for every relation pair."""
    nRel = len(cores)
    support = np.zeros((nRel, nRel), dtype=np.float32)
    bodyNnz = np.zeros((nRel, nRel), dtype=np.float32)
    for a in range(nRel):
        for b in range(nRel):
            path = cores[a] @ cores[b]                      # contraction over the shared middle variable
            body = (path > 0).astype(np.float32)            # existence semantics
            support[a, b] = head.multiply(body).sum()
            bodyNnz[a, b] = path.nnz
    with np.errstate(invalid="ignore", divide="ignore"):
        confidence = np.where(bodyNnz > 0, support / np.maximum(bodyNnz, 1), 0.0)
    confidence[support < minSupport] = 0.0
    return support, bodyNnz, confidence


def best_rules(relations, support, confidence, topk=5, minSupport=1, sortKey="confidence"):
    """Top-k relation pairs; ties on the primary key broken by the other one."""
    if sortKey == "confidence":
        order = np.lexsort((-support.ravel(), -confidence.ravel()))
    else:
        order = np.lexsort((-confidence.ravel(), -support.ravel()))
    out = []
    for flat in order:
        a, b = divmod(int(flat), len(relations))
        if support[a, b] < minSupport:
            break
        out.append((relations[a], relations[b], float(support[a, b]), float(confidence[a, b])))
        if len(out) == topk:
            break
    return out


def run_kg(path, topk=5, minSupport=5, gate=False):
    t0 = time.time()
    entities, relations, rows, cols, rIdx = load_triples(path)
    print("loaded {} triples, {} entities, {} relations ({:.1f}s)".format(
        len(rows), len(entities), len(relations), time.time() - t0), flush=True)
    cores = relation_cores(rows, cols, rIdx, len(entities), len(relations))
    results = {"confidence": {}, "support": {}}
    selfCells = {}
    for headIdx, headRel in enumerate(relations):
        support, bodyNnz, confidence = score_chains(cores, cores[headIdx], minSupport=minSupport)
        selfCells[headRel] = (int(support[headIdx, headIdx]), int(bodyNnz[headIdx, headIdx]))
        results["confidence"][headRel] = best_rules(relations, support, confidence, topk=topk,
                                                    minSupport=minSupport, sortKey="confidence")
        results["support"][headRel] = best_rules(relations, support, confidence, topk=topk,
                                                 minSupport=minSupport, sortKey="support")
        print("\n{}  ({} facts)".format(headRel, int(cores[headIdx].nnz)), flush=True)
        for tag in ("support", "confidence"):
            for b1, b2, s, c in results[tag][headRel]:
                print("    [{}] {}(X,Z) ^ {}(Z,Y) -> {}(X,Y)   support={:g}  conf={:.3f}".format(
                    tag, b1, b2, headRel, s, c), flush=True)
    print("\ntotal {:.1f}s".format(time.time() - t0), flush=True)
    if gate:
        check_gate(results, selfCells)
    return results


def check_gate(results, selfCells):
    """Gate on real-KG structure, plus the diagnostic that decides the benchmark."""
    top = results["support"]["_also_see"][0]
    assert (top[0], top[1]) == ("_also_see", "_also_see"), "_also_see self-rule not top by support, got {}".format(top[:2])
    print("gate: _also_see self-rule top by support (support={:g})".format(top[2]), flush=True)

    support, twoHop = selfCells["_hypernym"]
    assert twoHop > 10000 and support < 100, "hypernym transitivity diagnostic changed: support={}, twoHop={}".format(
        support, twoHop)
    print("diagnostic: hypernym self-rule covers {}/{} two-hop pairs -> graph not transitively closed".format(
        support, twoHop), flush=True)
    print("gate passed", flush=True)


def selfDiagnostics(path, topn=15):
    """Per relation: how much of its two-hop closure is observed as facts (r,r) -> r."""
    entities, relations, rows, cols, rIdx = load_triples(path)
    print("{}: {} triples, {} entities, {} relations".format(
        os.path.basename(os.path.dirname(path)) or path, len(rows), len(entities), len(relations)), flush=True)
    cores = relation_cores(rows, cols, rIdx, len(entities), len(relations))
    table = []
    for a, rel in enumerate(relations):
        path2 = cores[a] @ cores[a]
        twoHop = path2.nnz
        if twoHop == 0:
            continue
        support = cores[a].multiply((path2 > 0).astype(np.float32)).sum()
        table.append((rel, int(cores[a].nnz), int(support), twoHop, float(support) / twoHop))
    table.sort(key=lambda row: -row[4])
    print("{:<42} {:>8} {:>8} {:>10} {:>8}".format("relation", "facts", "r o r", "twoHop", "ratio"), flush=True)
    for rel, facts, support, twoHop, ratio in table[:topn]:
        print("{:<42} {:>8} {:>8} {:>10} {:>8.4f}".format(rel, facts, support, twoHop, ratio), flush=True)
    return table


def selftest():
    """Planted r3 = r1 o r2 must be the argmax for head r3."""
    rng = np.random.default_rng(0)
    nEnt, nRel = 40, 4
    cores = []
    for _ in range(nRel - 1):
        dense = (rng.random((nEnt, nEnt)) < 0.15).astype(np.float32)
        cores.append(sparse.csr_matrix(dense))
    planted = ((cores[0] @ cores[1]) > 0).astype(np.float32)
    flip = sparse.csr_matrix((rng.random((nEnt, nEnt)) < 0.01).astype(np.float32))
    planted = ((planted + flip) > 0).astype(np.float32)
    cores.append(sparse.csr_matrix(planted))

    relations = ["r0", "r1", "r2", "r3"]
    support, bodyNnz, confidence = score_chains(cores, cores[3], minSupport=1)
    best = np.unravel_index(np.argmax(confidence), confidence.shape)
    print("selftest argmax:", relations[best[0]], relations[best[1]],
          "conf={:.3f} support={:g}".format(confidence[best], support[best]), flush=True)
    assert best == (0, 1), "planted rule (r0,r1)->r3 not recovered, got {}".format(best)
    assert confidence[0, 1] > 0.9, "planted rule confidence too low: {}".format(confidence[0, 1])
    print("selftest passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--triples", default=os.path.join(HERE, "data", "wn18rr", "train.txt"))
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--minSupport", type=int, default=5)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--diag", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.diag:
        selfDiagnostics(args.triples)
    else:
        run_kg(args.triples, topk=args.topk, minSupport=args.minSupport, gate=args.gate)
