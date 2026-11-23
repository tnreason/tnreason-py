"""Why is the polynomial contractor twelve times slower than a sparse product?

Three candidate explanations, measured separately on real FB15k-237 relations:

  algorithmic  the pairwise monomial scan inspects m1*m2 candidate pairs to produce nnz matches
  constant     each inspected pair is Python-level dictionary work, not a compiled kernel
  representation  a polynomial materialises one monomial per witness, so the dense tail costs memory

Usage:
    python slow_probe.py [--dir data/fb15k237]
"""

import argparse
import os
import time

import numpy as np

from tnreason import engine

import kg_rules as K

HERE = os.path.dirname(os.path.abspath(__file__))


def poly_core(edges, colors, n, name):
    values = [(1.0, {colors[0]: int(x), colors[1]: int(y)}) for x, y in edges]
    return engine.get_core("PolynomialCore")(values=values, colors=colors, shape=[n, n], name=name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=os.path.join(HERE, "data", "fb15k237"))
    parser.add_argument("--pairs", type=int, default=12)
    args = parser.parse_args()

    ents, rels, rows, cols, rIdx = K.load_triples(os.path.join(args.dir, "train.txt"))
    n, nRel = len(ents), len(rels)
    head = K.relation_cores(rows, cols, rIdx, n, nRel)
    cores = head + [c.T.tocsr() for c in head]

    rng = np.random.default_rng(1)
    order = np.argsort([-c.nnz for c in cores])
    picks = [int(order[i]) for i in rng.choice(min(40, len(order)), args.pairs, replace=False)]
    picks.sort(key=lambda a: cores[a].nnz)

    print("entities {}, relation sizes {} to {}".format(
        n, cores[picks[0]].nnz, cores[picks[-1]].nnz))
    print("{:>8} {:>8} {:>10} {:>9} {:>11} {:>9} {:>8}".format(
        "m1", "m2", "pairs", "matches", "scipy ms", "poly ms", "ratio"))

    for a in picks:
        b = int(rng.integers(len(cores)))
        A, Bm = cores[a], cores[b]
        if A.nnz == 0 or Bm.nnz == 0:
            continue
        t0 = time.perf_counter()
        prod = (A @ Bm).tocsr()
        tScipy = (time.perf_counter() - t0) * 1e3

        ax, ay = np.nonzero(A)
        bx, by = np.nonzero(Bm)
        coreL = poly_core(np.stack([ax, ay], axis=1), ["X", "Z"], n, "l")
        coreR = poly_core(np.stack([bx, by], axis=1), ["Z", "Y"], n, "r")
        t0 = time.perf_counter()
        res = engine.contract(coreDict={"l": coreL, "r": coreR}, openColors=["X", "Y"],
                              contractionMethod="CorewiseContractor")
        tPoly = (time.perf_counter() - t0) * 1e3

        witnesses = sum(1 for v, _ in res.values if v != 0)
        print("{:>8} {:>8} {:>10.0f} {:>9} {:>11.2f} {:>9.1f} {:>8.1f}".format(
            A.nnz, Bm.nnz, A.nnz * Bm.nnz, prod.nnz, tScipy, tPoly,
            tPoly / max(tScipy, 1e-6)))

    print("\npairs scanned against matches produced is the algorithmic waste;")
    print("scipy spends its time inside a compiled hash join, the polynomial arm inside Python dicts.")


if __name__ == "__main__":
    main()
