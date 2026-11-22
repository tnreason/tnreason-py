"""Numerical comparison as an operator, not a matrix.

A comparison factor over entities, M[z,y] = 1[p_z < q_y], is dense by nature and must never be built.
It is applied instead as a cumulative sum: sort the left entities by their feature, take running sums of
the weight column, and read off the value at the count of entities below each threshold. The cost is
dominated by the sort, against n^2 entries for the materialised factor.

This is the construction used for numerical rules in the differentiable rule-learning literature, here
as an operator inside a contraction: the factor participates in a body exactly like a relational atom,
but it is applied rather than stored.
"""

import numpy as np
from scipy import sparse


class ComparisonOperator:
    """M[z, y] = 1[p[z] OP q[y]] for a bracket operator, applied by cumulative sums.

    Only the two-sided order is needed for `<` and `>`; equality and inequality fall back to a direct
    construction because they are not monotone.
    """

    def __init__(self, leftValues, rightValues, operator="<", slope=1.0, intercept=0.0,
                 halfWidth=None):
        self.left = np.asarray(leftValues, dtype=np.float64)
        self.right = np.asarray(rightValues, dtype=np.float64)
        self.operator = operator
        self.slope = float(slope)
        self.intercept = float(intercept)
        self.halfWidth = halfWidth
        if operator not in ("<", ">", "<=", ">="):
            raise NotImplementedError("operator {} is not monotone".format(operator))
        if self.slope == 0.0:
            raise ValueError("a zero slope is not a comparison between the two sides")

    def __len__(self):
        return self.left.size

    def apply(self, weightMatrix, leftIsRows=True):
        """A @ M for a sparse A whose columns run over the left entities of the comparison."""
        weights = weightMatrix.tocsc() if sparse.issparse(weightMatrix) else sparse.csc_matrix(weightMatrix)
        if weights.shape[1] != self.left.size:
            raise ValueError("weight columns {} against {} left values".format(
                weights.shape[1], self.left.size))

        # an affine map of one side keeps the comparison monotone; a negative slope flips it
        left = self.slope * self.left + self.intercept
        right = self.right
        operator = self.operator
        if self.slope < 0:
            operator = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}[operator]
        if operator in (">", ">="):
            left = -left
            right = -right
            strict = operator == ">"
        else:
            strict = operator == "<"

        order = np.argsort(left, kind="stable")
        sortedLeft = left[order]
        if self.halfWidth is None:
            counts = np.searchsorted(sortedLeft, right, side="left" if strict else "right")
            lowCounts = np.zeros_like(counts)
        else:
            # a band is the difference of two prefix sums on the same sorted order
            counts = np.searchsorted(sortedLeft, right + self.halfWidth, side="right")
            lowCounts = np.searchsorted(sortedLeft, right - self.halfWidth, side="left")
        safeHigh = np.maximum(counts - 1, 0)
        safeLow = np.maximum(lowCounts - 1, 0)

        result = np.zeros((weights.shape[0], right.size), dtype=np.float64)
        for row in range(weights.shape[0]):
            rowWeights = np.asarray(weights.getrow(row).todense()).ravel()[order]
            running = np.cumsum(rowWeights)
            high = np.where(counts > 0, running[safeHigh], 0.0)
            if self.halfWidth is None:
                result[row] = high
            else:
                low = np.where(lowCounts > 0, running[safeLow], 0.0)
                result[row] = high - low
        return sparse.csr_matrix(result)


def select_affine(leftValues, rightValues, weights, truth, slopes=(0.5, 1.0, 2.0, 4.0),
                  intercepts=None, chunk=9):
    """Grid over slope and intercept for a condition alpha*x + beta < y, reusing one sort per slope.

    The intercept only shifts the threshold, so the sorted order is the same for every intercept at a
    given slope: one sort and one set of running sums serve the whole row of the grid. The criterion is
    the mass placed on known pairs minus the mass placed elsewhere, which is what a rule learner would
    accept on.
    """
    weights = weights.tocsc()
    dense = np.asarray(weights.todense(), dtype=np.float64)
    truthDense = np.asarray(truth.todense() if sparse.issparse(truth) else truth, dtype=bool)
    if intercepts is None:
        span = float(np.abs(rightValues).max() - np.abs(rightValues).min()) or 1.0
        intercepts = np.linspace(-span, span, chunk)

    best, bestScore = None, -np.inf
    for slope in slopes:
        order = np.argsort(slope * leftValues, kind="stable")
        sortedKey = (slope * leftValues)[order]
        running = np.cumsum(dense[:, order], axis=1)
        for intercept in intercepts:
            counts = np.searchsorted(sortedKey, rightValues - intercept, side="left")
            safe = np.maximum(counts - 1, 0)
            values = np.where(counts > 0, running[:, safe], 0.0)
            score = float(values[truthDense].sum() - values[~truthDense].sum())
            if score > bestScore:
                best, bestScore = (float(slope), float(intercept)), score
    return {"slope": best[0], "intercept": best[1], "score": bestScore}


def materialized(leftValues, rightValues, operator="<"):
    """The dense factor, for tests only."""
    left = np.asarray(leftValues, dtype=np.float64)[:, None]
    right = np.asarray(rightValues, dtype=np.float64)[None, :]
    return {
        "<": left < right, ">": left > right,
        "<=": left <= right, ">=": left >= right,
    }[operator].astype(np.float64)


def _self_check():
    rng = np.random.default_rng(0)
    n = 25
    left = rng.integers(0, 40, size=n).astype(np.float64)
    right = rng.integers(0, 40, size=n).astype(np.float64)
    weights = sparse.random(6, n, density=0.4, random_state=1, format="csr")

    for operator in ("<", ">", "<=", ">="):
        fast = ComparisonOperator(left, right, operator).apply(weights)
        dense = weights.toarray() @ materialized(left, right, operator)
        difference = np.abs(fast.toarray() - dense).max()
        assert difference < 1e-10, "{} disagrees: {}".format(operator, difference)
        print("operator {}: matches the materialised factor, max difference {:.1e}".format(
            operator, difference))

    # an affine transform of one side is still a monotone comparison, so it stays a sort
    for alpha, beta in ((2.0, 3.0), (0.5, -1.0)):
        for operator in ("<", ">"):
            scaled = ComparisonOperator(alpha * left + beta, right, operator).apply(weights)
            dense = weights.toarray() @ materialized(alpha * left + beta, right, operator)
            difference = np.abs(scaled.toarray() - dense).max()
            assert difference < 1e-10, "affine {} {} disagrees: {}".format(alpha, operator, difference)
            print("affine alpha={} beta={} operator {}: matches to {:.1e}".format(
                alpha, beta, operator, difference))

    # a genuinely two-sided comparison is not a sort on either side: it needs a 2-D structure
    a2 = rng.random((n, 2))
    orderByFirst = np.argsort(a2[:, 0])
    blocks = np.searchsorted(a2[orderByFirst, 0], a2[:, 0])
    violation = 0
    for i in range(n):
        for j in range(n):
            if a2[i].sum() < a2[j].sum() and orderByFirst[blocks[i]] > orderByFirst[blocks[j]]:
                violation += 1
    print("two-sided sums: {} pairs where a 1-D sort on one column gives the wrong order".format(
        violation))

    # a band is the difference of two prefix sums on the same sort
    for halfWidth in (2.0, 5.0):
        fast = ComparisonOperator(left, right, "<", halfWidth=halfWidth).apply(weights)
        band = np.abs(left[:, None] - right[None, :]) <= halfWidth
        dense = weights.toarray() @ band.astype(np.float64)
        difference = np.abs(fast.toarray() - dense).max()
        assert difference < 1e-10, "band {} disagrees: {}".format(halfWidth, difference)
        print("band halfWidth={}: matches to {:.1e}".format(halfWidth, difference))

    # a negative slope flips the comparison and must still be exact
    flipped = ComparisonOperator(left, right, "<", slope=-2.0).apply(weights)
    dense = weights.toarray() @ materialized(-2.0 * left, right, ">")
    difference = np.abs(flipped.toarray() - dense).max()
    assert difference < 1e-10, "negative slope disagrees: {}".format(difference)
    print("negative slope: matches to {:.1e}".format(difference))

    # the grid must recover a planted slope and intercept from the same one-sort sweep
    plantedSlope, plantedIntercept = 2.0, 6.0
    rows = rng.choice(n, size=6, replace=False)
    oneHot = np.zeros((6, n))
    oneHot[np.arange(6), rows] = 1.0
    truth = sparse.csr_matrix(((plantedSlope * left[rows, None] + plantedIntercept) < right[None, :])
                              .astype(np.float64))
    picked = select_affine(left, right, sparse.csr_matrix(oneHot), truth)
    # the parameters are identified only up to the partition they induce on the observed values
    replayed = ComparisonOperator(left, right, "<", slope=picked["slope"],
                                  intercept=picked["intercept"]).apply(sparse.csr_matrix(oneHot))
    recovered = np.array_equal(replayed.toarray() > 0, truth.toarray() > 0)
    print("grid search: picked slope {} intercept {:.2f} against planted {} and {}".format(
        picked["slope"], picked["intercept"], plantedSlope, plantedIntercept))
    assert recovered, "grid did not recover the planted configuration"

    full = 40 * 40
    stored = weights.nnz
    print("materialising the factor would store {} entries per body; applying it stores {}".format(
        full, stored))


if __name__ == "__main__":
    _self_check()
