"""The pair scan is the whole cost, and an index removes it.

The polynomial contractor compares every monomial of one core against every monomial of the other and
keeps the pairs whose shared colour agrees. On real FB15k-237 relations that is up to 22 million
comparisons producing zero matches. A hash join on the shared colour produces the same answer in
time proportional to the input plus the output, which is what a sparse matrix product does.

This script measures the aggregate cost over the bodies the learner actually prices, then times an
indexed join over the same bodies and checks that the support is identical.

Usage:
    python indexed_probe.py [--dir data/fb15k237] [--sample 400]
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np

import boost as B
import kg_rules as K

HERE = os.path.dirname(os.path.abspath(__file__))


def indexed_product(left, right):
    """(x,z) pairs joined with (z,y) pairs on z: cost is input plus output, not input squared."""
    byMiddle = defaultdict(list)
    rows, cols = right                    # rows are the shared colour, cols the open one
    for z, y in zip(rows.tolist(), cols.tolist()):
        byMiddle[z].append(y)
    out = []
    for x, z in zip(left[0].tolist(), left[1].tolist()):
        for y in byMiddle.get(z, ()):
            out.append((x, y))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=os.path.join(HERE, "data", "fb15k237"))
    parser.add_argument("--sample", type=int, default=400)
    args = parser.parse_args()

    ents, rels, rows, cols, rIdx = K.load_triples(os.path.join(args.dir, "train.txt"))
    n, nRel = len(ents), len(rels)
    head = K.relation_cores(rows, cols, rIdx, n, nRel)
    cores = head + [c.T.tocsr() for c in head]
    nAtom = len(cores)
    dom, rang = B.domains_ranges(cores)
    joint = np.array([B.overlaps(rang[a], dom) for a in range(nAtom)])

    edges = []
    for c in cores:
        xs, zs = np.nonzero(c)
        edges.append((xs, zs))

    viable = [(a, int(b)) for a in range(nAtom) for b in np.nonzero(joint[a])[0]]
    pairWork = 0.0
    for a, b in viable:
        pairWork += float(cores[a].nnz) * float(cores[b].nnz)

    rate = 20.8e6            # pair comparisons per second, measured in slow_probe
    print("viable bodies {}".format(len(viable)))
    print("total monomial pairs to compare: {:.3e}".format(pairWork))
    print("at {:.1e} comparisons/s the polynomial scan needs {:.0f} s for the whole space".format(
        rate, pairWork / rate))

    rng = np.random.default_rng(2)
    sample = [viable[i] for i in rng.choice(len(viable), min(args.sample, len(viable)), replace=False)]

    t0 = time.perf_counter()
    for a, b in sample:
        (cores[a] @ cores[b]).tocsr()
    tScipy = time.perf_counter() - t0

    t0 = time.perf_counter()
    totalPairs = 0
    for a, b in sample:
        out = indexed_product(edges[a], edges[b])
        totalPairs += len(out)
    tIndexed = time.perf_counter() - t0

    print("\non {} sampled bodies: scipy {:.3f}s ({:.0f}/s), indexed join {:.3f}s ({:.0f}/s), ratio {:.1f}x".format(
        len(sample), tScipy, len(sample) / tScipy, tIndexed, len(sample) / tIndexed, tIndexed / tScipy))
    print("output pairs produced: {:.3e}".format(totalPairs))

    # correctness on a handful of bodies: the join must produce the same support as the product
    for a, b in sample[:20]:
        joined = set(indexed_product(edges[a], edges[b]))
        sparseSupport = set(zip(*np.nonzero((cores[a] @ cores[b]).tocsr() > 0)))
        if joined != sparseSupport:
            raise SystemExit("indexed join disagrees with the sparse product on {} then {}".format(a, b))
    print("support identical on 20 checked bodies")


if __name__ == "__main__":
    main()
