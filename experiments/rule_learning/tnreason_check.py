"""Does the sparse learner compute the same thing as a tnreason contraction?

The paper claims that a body score is a contraction. The learner never imported tnreason, so the
claim is currently asserted rather than tested. This script builds the same three body shapes as
tnreason cores over real Kinship adjacencies, contracts them with the tnreason engine, and compares
the resulting score tensor against the sparse products in boost.chain.

Chain   r1(X,Z) r2(Z,Y)              -> contract {r1 on X,Z; r2 on Z,Y} open (X,Y)
Tree    r1(X,Z) r2(Z,Y) c(X,W)       -> the same plus a gate core on X, opened on (X,Y)
Fork    r1(X,Z) r2(Y,Z)              -> r2 transposed on Y,Z, which is the chain over the inverse

Usage:
    python tnreason_check.py [--dir data/kinship]
"""

import argparse
import os

import numpy as np
from scipy import sparse

from tnreason import engine

import boost as B
import kg_rules as K


def dense(mat):
    if not sparse.issparse(mat) and hasattr(mat, "values"):     # tnreason Tensor wraps an ndarray
        mat = mat.values
    out = mat.toarray() if sparse.issparse(mat) else np.asarray(mat)
    return np.asarray(out, dtype=np.float64)


def as_core(values, colors, name):
    # NumpyCore wants a dense ndarray; a sparse operand makes the contractor emit the wrong subscript count
    return engine.get_core()(values=dense(values), colors=colors, name=name)


def compare(label, tn, sparseMat, tol=1e-8):
    """Compare on existential support first, then report how far the raw values are apart.

    The learner defines a body feature as existential truth: X and Y are joined by the body when at
    least one witness exists. A raw contraction instead counts witnesses, so the two agree on support
    and can differ in magnitude wherever a body has several witnesses. Support agreement is the
    equivalence the learner needs; the magnitude gap is what a port has to decide about.
    """
    tnDense = dense(tn)
    spDense = dense(sparseMat)
    if tnDense.shape != spDense.shape:
        raise SystemExit("{}: shape {} against {}".format(label, tnDense.shape, spDense.shape))
    supportAgrees = bool(((tnDense > 0) == (spDense > 0)).all())
    diff = np.abs(tnDense - spDense)
    exact = bool(diff.max() <= tol)
    print("{}: shape {}  support identical {}  max abs difference {:.3e}  value-identical {}".format(
        label, tnDense.shape, supportAgrees, diff.max(), exact))
    return supportAgrees


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "data", "kinship"))
    args = parser.parse_args()

    entities, relations, rows, cols, rIdx = K.load_triples(os.path.join(args.dir, "train.txt"))
    n, nRel = len(entities), len(relations)
    headCores = K.relation_cores(rows, cols, rIdx, n, nRel)
    cores = headCores + [c.T.tocsr() for c in headCores]          # atoms, then inverse atoms

    def pick(predicate):
        for r in range(nRel):
            if predicate(relations[r]) and headCores[r].nnz > 10:
                return r
        return None

    r1 = pick(lambda name: True)
    r2 = pick(lambda name: name != relations[r1])
    r3 = pick(lambda name: name not in (relations[r1], relations[r2]))
    print("relations used: {} ({} edges), {} ({} edges), gate {} ({} edges)".format(
        relations[r1], headCores[r1].nnz, relations[r2], headCores[r2].nnz,
        relations[r3], headCores[r3].nnz))

    A1 = headCores[r1]
    A2 = headCores[r2]
    # the gate must be an independent relation: taken from r1 it is implied by the chain itself
    gate = np.asarray((headCores[r3].getnnz(axis=1) > 0)).ravel().astype(np.float64)
    print("gate holds for {} of {} heads".format(int(gate.sum()), gate.size))

    # chain: X -> Z -> Y
    tnChain = engine.contract(coreDict={"r1": as_core(A1, ["X", "Z"], "r1"),
                                        "r2": as_core(A2, ["Z", "Y"], "r2")},
                              openColors=["X", "Y"])
    ok1 = compare("chain  contraction vs sparse product", tnChain, B.chain(cores, (r1, r2)))

    # tree: the same chain, gated on X by an existential premise c(X,W)
    tnTree = engine.contract(coreDict={"r1": as_core(A1, ["X", "Z"], "r1"),
                                       "r2": as_core(A2, ["Z", "Y"], "r2"),
                                       "c": as_core((headCores[r3] > 0).astype(np.float64), ["X", "W"], "c")},
                             openColors=["X", "Y"])
    ok2 = compare("tree   contraction vs masked product", tnTree, B.chain(cores, ("tx", r1, r2, r3)))

    # fork: the second premise is traversed backwards, which is the chain over the inverse atom
    tnFork = engine.contract(coreDict={"r1": as_core(A1, ["X", "Z"], "r1"),
                                       "r2": as_core(A2, ["Y", "Z"], "r2")},
                             openColors=["X", "Y"])
    ok3 = compare("fork   contraction vs sparse product", tnFork, B.chain(cores, ("fz", r1, r2)))

    print("supports agree on all three shapes" if all((ok1, ok2, ok3))
          else "MISMATCH: the sparse learner is not the contraction")


if __name__ == "__main__":
    main()
