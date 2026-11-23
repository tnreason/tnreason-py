"""P1' of the port plan: can any tnreason core type price a body at FB15k-237 scale?

The survey walks the three contractable core types that are actually available in this environment and
measures how their contraction cost grows. It reports the fitted exponent and the extrapolated cost of
one two-literal body at FB15k-237 sizes, against the requirement of 200 bodies per second, i.e. 5 ms
per body. HypertrieCore is listed but cannot be exercised, since the tentris package is not installed.

Usage:
    python coretype_survey.py [--dir data/fb15k237]
"""

import argparse
import os
import time

import numpy as np

from tnreason import engine

HERE = os.path.dirname(os.path.abspath(__file__))


def timeit(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def fit_exponent(sizes, times):
    slope, _ = np.polyfit(np.log(sizes), np.log(times), 1)
    return slope


def polynomial_core_from_edges(edges, n, colors, name):
    """One monomial per edge: the cheapest sparse representation the polynomial core offers."""
    values = [(1.0, {colors[0]: int(x), colors[1]: int(y)}) for x, y in edges]
    return engine.get_core("PolynomialCore")(values=values, colors=colors, shape=[n, n], name=name)


def survey_polynomial(n, monomials, seed=0):
    rng = np.random.default_rng(seed)
    sizes, times = [], []
    for m in monomials:
        a = rng.integers(0, n, size=(m, 2))
        b = rng.integers(0, n, size=(m, 2))
        coreA = polynomial_core_from_edges(a, n, ["X", "Z"], "a")
        coreB = polynomial_core_from_edges(b, n, ["Z", "Y"], "b")
        # polynomial cores have no einsum view; they contract corewise, pairwise over monomials
        elapsed = timeit(lambda: engine.contract(coreDict={"a": coreA, "b": coreB},
                                                 openColors=["X", "Y"],
                                                 contractionMethod="CorewiseContractor"), repeats=1)
        sizes.append(m)
        times.append(elapsed)
        print("  polynomial: {} monomials per core -> {:.3f}s".format(m, elapsed), flush=True)
    return np.array(sizes), np.array(times)


def survey_numpy(nEnts, seed=0):
    rng = np.random.default_rng(seed)
    sizes, times = [], []
    for n in nEnts:
        # 0.1 per cent density, close to FB15k-237's 0.13 per cent
        a = (rng.random((n, n)) < 0.001).astype(np.float64)
        b = (rng.random((n, n)) < 0.001).astype(np.float64)
        coreA = engine.get_core()(values=a, colors=["X", "Z"], name="a")
        coreB = engine.get_core()(values=b, colors=["Z", "Y"], name="b")
        elapsed = timeit(lambda: engine.contract(coreDict={"a": coreA, "b": coreB},
                                                 openColors=["X", "Y"]), repeats=2)
        sizes.append(n)
        times.append(elapsed)
        print("  numpy dense: {} entities -> {:.4f}s".format(n, elapsed), flush=True)
    return np.array(sizes), np.array(times)


def correctness_check(kinshipDir):
    """Do polynomial cores contract to the same support as the sparse product on real data?"""
    import boost as B
    import kg_rules as K

    ents, rels, rows, cols, rIdx = K.load_triples(os.path.join(kinshipDir, "train.txt"))
    n, nRel = len(ents), len(rels)
    head = K.relation_cores(rows, cols, rIdx, n, nRel)
    cores = head + [c.T.tocsr() for c in head]

    def edges_of(core):
        xs, ys = np.nonzero(core)
        return np.stack([xs, ys], axis=1)

    checked = 0
    for a in range(min(len(cores), 12)):
        for b in range(min(len(cores), 12)):
            if cores[a].nnz == 0 or cores[b].nnz == 0:
                continue
            coreA = polynomial_core_from_edges(edges_of(cores[a]), n, ["X", "Z"], "a")
            coreB = polynomial_core_from_edges(edges_of(cores[b]), n, ["Z", "Y"], "b")
            tn = engine.contract(coreDict={"a": coreA, "b": coreB}, openColors=["X", "Y"],
                                 contractionMethod="CorewiseContractor")
            # a polynomial contraction returns monomials, not an array: read the open-colour support
            tnPairs = {(int(pos["X"]), int(pos["Y"])) for val, pos in tn.values
                       if val != 0 and "X" in pos and "Y" in pos}
            sparseSupport = {(int(x), int(y)) for x, y in
                             zip(*np.nonzero(B.chain(cores, (a, b)) > 0))}
            if tnPairs != sparseSupport:
                print("  MISMATCH for atoms {} then {}".format(a, b))
                return False
            checked += 1
    print("  polynomial support agrees with the sparse product on {} body pairs".format(checked))
    return checked > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=os.path.join(HERE, "data", "fb15k237"))
    args = parser.parse_args()

    requirement = 1.0 / 200.0
    print("requirement for 200 bodies/s: {:.3f} s per body".format(requirement))

    print("\nPolynomialCore, sparse, one monomial per edge:")
    pSizes, pTimes = survey_polynomial(2000, [100, 200, 400, 800])
    pExp = fit_exponent(pSizes, pTimes)
    print("  fitted exponent {:.2f} (2.0 means pairwise monomial scan)".format(pExp))

    medianEdges = 1148          # 272 115 edges over 237 relations
    predictedMedian = pTimes[-1] * (medianEdges / pSizes[-1]) ** pExp
    print("  one body at the median FB15k-237 relation size ({} edges): {:.3f}s".format(
        medianEdges, predictedMedian))

    print("\nNumpyCore, dense:")
    nSizes, nTimes = survey_numpy([500, 1000, 2000, 4000])
    nExp = fit_exponent(nSizes, nTimes)
    print("  fitted exponent {:.2f} (3.0 means dense matrix product)".format(nExp))
    predictedFull = nTimes[-1] * (14505 / nSizes[-1]) ** nExp
    print("  one body at 14 505 entities: {:.1f}s".format(predictedFull))
    print("  one dense core at 14 505 entities: {:.2f} GB, times 474 atoms = {:.0f} GB".format(
        14505 ** 2 * 8 / 1e9, 474 * 14505 ** 2 * 8 / 1e9))

    kinshipDir = os.path.join(HERE, "data", "kinship")
    if os.path.exists(kinshipDir):
        print("\nPolynomialCore correctness on Kinship:")
        correctness_check(kinshipDir)

    print("\nHypertrieCore: unavailable, the tentris package is not installed in this environment.")

    print("\nverdict")
    for label, perBody in (("polynomial sparse", predictedMedian), ("dense", predictedFull)):
        print("  {:18s} {:.3f}s per body = {:.1f} bodies/s, needed {:.0f}/s, short by {:.0f}x".format(
            label, perBody, 1.0 / perBody, 200, perBody / requirement))


if __name__ == "__main__":
    main()
