import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.constraint_networks import forward_chaining as fc
from experiments.constraint_networks import forward_chaining_legacy as legacy_fc
from tnreason import engine


def _example_core_dict():
    return {
        "a_bas": engine.create_from_slice_iterator(
            colors=["a"], shape=[2], sliceIterator=[(1, {"a": 0})]
        ),
        "ab_ent": engine.create_from_slice_iterator(
            colors=["a", "b"],
            shape=[2, 3],
            sliceIterator=[
                (1, {"a": 0, "b": 0}),
                (1, {"a": 1, "b": 1}),
                (1, {"a": 1, "b": 2}),
            ],
        ),
        "bc_ent": engine.create_from_slice_iterator(
            colors=["c", "b"],
            shape=[2, 3],
            sliceIterator=[
                (1, {"c": 1, "b": 0}),
                (1, {"c": 0, "b": 1}),
            ],
        ),
    }


def _branching_core_dict():
    return {
        "a_bas": engine.create_from_slice_iterator(
            colors=["a"], shape=[2], sliceIterator=[(1, {"a": 0})]
        ),
        "ab_ent": engine.create_from_slice_iterator(
            colors=["a", "b"],
            shape=[2, 2],
            sliceIterator=[(1, {"a": 0, "b": 0}), (1, {"a": 1, "b": 1})],
        ),
        "ac_ent": engine.create_from_slice_iterator(
            colors=["a", "c"],
            shape=[2, 2],
            sliceIterator=[(1, {"a": 0, "c": 1}), (1, {"a": 1, "c": 0})],
        ),
    }


def test_legacy_forward_chaining_remains_importable():
    assert legacy_fc.GenericForwardChaining is not fc.GenericForwardChaining


def test_forward_chaining_rebuilt_indices_have_no_stale_edges():
    chainer = fc.GenericForwardChaining(_example_core_dict())
    chainer.propagate_all_singleNodeEdges()
    chainer.validate_indices()

    for edge_keys in chainer.nodesDict.values():
        assert len(edge_keys) == len(set(edge_keys))
        assert set(edge_keys) <= set(chainer.cn.coresDict)

    for edge_keys in chainer.singleNodeEdges.values():
        assert len(edge_keys) == len(set(edge_keys))
        assert set(edge_keys) <= set(chainer.cn.coresDict)


def test_forward_chaining_keeps_original_example_behavior():
    chainer = fc.GenericForwardChaining(_example_core_dict())
    chainer.propagate_all_singleNodeEdges()

    assert chainer.cn.coresDict["a_core"][{"a": 0}] == 1
    assert chainer.cn.coresDict["a_core"][{"a": 1}] == 0
    assert chainer.cn.coresDict["b_core"][{"b": 0}] == 1
    assert chainer.cn.coresDict["b_core"][{"b": 1}] == 0
    c_edge = chainer.singleNodeEdges["c"][0]
    assert chainer.cn.coresDict[c_edge][{"c": 0}] == 0
    assert chainer.cn.coresDict[c_edge][{"c": 1}] == 1


def test_parallel_and_serial_forward_chaining_match():
    serial_chainer = fc.GenericForwardChaining(_branching_core_dict(), parallel=False)
    start_time = time.perf_counter()
    serial_chainer.propagate_all_singleNodeEdges()
    serial_seconds = time.perf_counter() - start_time

    parallel_chainer = fc.GenericForwardChaining(_branching_core_dict(),parallel=True,max_workers=8)
    start_time = time.perf_counter()
    parallel_chainer.propagate_all_singleNodeEdges()
    parallel_seconds = time.perf_counter() - start_time

    print(
        "forward chaining timings: "
        f"serial={serial_seconds:.6f}s "
        f"parallel={parallel_seconds:.6f}s ",
        flush=True,
    )

    assert serial_chainer.consistent == parallel_chainer.consistent
    assert set(serial_chainer.cn.coresDict) == set(parallel_chainer.cn.coresDict)
    for core_key in serial_chainer.cn.coresDict:
        assert engine.cores_equal(
            serial_chainer.cn.coresDict[core_key],
            parallel_chainer.cn.coresDict[core_key],
        )


if __name__ == "__main__":
    test_parallel_and_serial_forward_chaining_match()
