"""How dense do the learner's real bodies get, and can any engine store them?

Measured on the bodies the learner actually prices on FB15k-237: two-literal bodies whose ranges and
domains overlap. Reports product nnz against the dense matrix, and the total footprint if every
viable product is held in memory, which is what the current learner does.
"""

import os

import numpy as np

import boost as B
import kg_rules as K

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ents, rels, rows, cols, rIdx = K.load_triples(os.path.join(HERE, "data", "fb15k237", "train.txt"))
    n, nRel = len(ents), len(rels)
    head = K.relation_cores(rows, cols, rIdx, n, nRel)
    cores = head + [c.T.tocsr() for c in head]
    nAtom = len(cores)
    dom, rang = B.domains_ranges(cores)
    joint = np.array([B.overlaps(rang[a], dom) for a in range(nAtom)])
    print("{} entities, {} relations, {} atoms, dense cell count {:.2e}".format(
        n, nRel, nAtom, n * n), flush=True)

    names = list(rels) + [r + "_inv" for r in rels]
    viable = []
    for a in range(nAtom):
        for b in np.nonzero(joint[a])[0]:
            prod = (cores[a] @ cores[int(b)]).tocsr()
            if prod.nnz:
                viable.append((a, int(b), int(prod.nnz)))

    nnz = np.array([v[2] for v in viable])
    print("viable two-literal bodies with nonempty product: {}".format(len(viable)))
    print("product nnz: median {:.0f}  p90 {:.0f}  max {:.0f}".format(
        np.median(nnz), np.percentile(nnz, 90), nnz.max()))
    print("as fraction of dense: median {:.6f}  p90 {:.6f}  max {:.6f}".format(
        np.median(nnz) / (n * n), np.percentile(nnz, 90) / (n * n), nnz.max() / (n * n)))
    print("holding every viable product: {:.2e} nonzeros, {:.2f} GB as float64".format(
        nnz.sum(), nnz.sum() * 8 / 1e9))
    for a, b, k in sorted(viable, key=lambda v: -v[2])[:5]:
        print("  densest: {} then {} -> {} pairs".format(names[a], names[b], k))


if __name__ == "__main__":
    main()
