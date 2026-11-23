"""Regressions for input work bounds found during independent review."""
import numpy as np
import pytest

from demonstrations.sudoku.inferenceClusters.core import RuleContext, RuleLimits, contract_cluster
from experiments.constraint_networks.sudoku_tests.standard_constraints import get_sudoku_constraint_network
from tnreason.engine.polynomial_handling import PolynomialCore
from tnreason.engine.workload_to_pandas import PandasCore


def test_unused_huge_sparse_dimension_is_rejected_before_domain_materialization():
    cores = get_sudoku_constraint_network(num=2)
    cores["unused_sparse"] = PolynomialCore(values=[(1, {})], colors=["unused"], shape=[10**9])
    with pytest.raises(ValueError, match="max_domain_values"):
        RuleContext(cores, num=2, limits=RuleLimits(max_domain_values=400))


def test_infinite_supplied_domain_has_a_bounded_read_limit():
    limit, reads = 400, []

    def unending():
        while True:
            reads.append(1)
            if len(reads) > limit + 1:
                pytest.fail("Domain normalization read beyond its finite budget")
            yield 0

    with pytest.raises(ValueError, match="max_domain_values"):
        RuleContext(get_sudoku_constraint_network(num=2),
                    {"pos_0_0_0_0": unending()}, num=2,
                    limits=RuleLimits(max_domain_values=limit))
    assert len(reads) <= limit + 1


def test_sparse_slice_scans_are_charged_before_realized_entry_reads(monkeypatch):
    cores = get_sudoku_constraint_network(num=2)
    key = "pos_0_0_0_0_a_0_0_0_0_0"
    original = cores[key]
    slices = [(1, {}), (-1, {})] * 1000
    cores[key] = PolynomialCore(values=slices, colors=original.colors, shape=original.shape)
    context = RuleContext(cores, num=2, limits=RuleLimits(max_work=100))
    cluster = context.describe("sparse_budget", (context.pos(0, 0),),
                               atoms=(context.atom(0, 0, 0),))
    monkeypatch.setattr(PolynomialCore, "__getitem__",
                        lambda *args: pytest.fail("Sparse getter ran before the work-budget check"))
    result = contract_cluster(context, cluster)
    assert result.status == "skipped_budget" and "work" in result.reason
    assert not result.summaries
    assert cores[key].values == slices


def test_pandas_realized_cancellation_and_iterator_are_preserved():
    cores = get_sudoku_constraint_network(num=2)
    key = "pos_0_0_0_0_a_0_0_0_0_0"
    original = cores[key]
    rows = [{**dict(zip(original.colors, index)), "values": int(original.values[index])}
            for index in np.ndindex(tuple(original.shape)) if original.values[index] != 0]
    # A pair of overlapping slices cancels at an otherwise unsupported entry.
    cancelled = {"a_0_0_0_0_0": 0, "pos_0_0_0_0": 0}
    rows.extend(({**cancelled, "values": 7}, {**cancelled, "values": -7}))
    sparse = PandasCore(values=rows, colors=original.colors, shape=original.shape)
    sparse.index = 1
    before = sparse.values.copy(deep=True)
    cores[key] = sparse
    context = RuleContext(cores, num=2)
    cluster = context.describe("pandas_support", (context.pos(0, 0),),
                               atoms=(context.atom(0, 0, 0),),
                               outputs=(context.pos(0, 0), context.atom(0, 0, 0)))
    result = contract_cluster(context, cluster, joint=True)
    assert result.status == "executed"
    summary = next(iter(result.summaries.values()))
    permutation = tuple(original.colors.index(color) for color in cluster.open_colors)
    np.testing.assert_array_equal(summary.values, (original.values != 0).transpose(permutation))
    assert sparse.index == 1 and sparse.values.equals(before) and sparse[cancelled] == 0
