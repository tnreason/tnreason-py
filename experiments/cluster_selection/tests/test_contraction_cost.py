from types import SimpleNamespace
import numpy as np
import pytest
from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from experiments.cluster_selection.contraction_cost import estimate_candidate_cost


def _core(colors, shape, value=1):
    return SimpleNamespace(colors=tuple(colors), shape=tuple(shape), values=np.full(shape, value))


def _candidate(keys, output):
    return SummaryClusterCandidate("test", tuple(keys), tuple(output))


def _contract(cores, candidate, path=None):
    colors = dict.fromkeys(color for key in candidate.core_keys for color in cores[key].colors)
    idx = {color: pos for pos, color in enumerate(colors)}
    args = []
    for key in candidate.core_keys:
        args.extend((cores[key].values, [idx[color] for color in cores[key].colors]))
    args.append([idx[color] for color in candidate.open_colors])
    return np.einsum(*args, optimize=False if path is None else ["einsum_path", *path])


@pytest.mark.parametrize("output", [("a", "b", "c"), ("c", "a"), ()])
def test_shared_hyperindex_is_preserved_and_path_matches_numpy(output):
    cores = {color: _core(("x", color), (2, 3), value=pos + 1)
             for pos, color in enumerate("abc")}
    candidate = _candidate(cores, output)
    estimate = estimate_candidate_cost(cores, candidate)
    np.testing.assert_array_equal(_contract(cores, candidate, estimate.path), _contract(cores, candidate))
    assert estimate.output_size == 3 ** len(output)
    if len(output) == 3:
        assert estimate.max_size == 27 > sum(core.values.size for core in cores.values())
        assert estimate.work == 72


def test_actual_mixed_shapes_not_nonzero_mask_sizes_are_used():
    cores = {"ab": _core(("a", "b"), (9, 2)), "bc": _core(("b", "c"), (2, 9))}
    cores["ab"].values[:] = 0
    candidate = _candidate(cores, ("a", "c"))
    estimate = estimate_candidate_cost(cores, candidate)
    assert (estimate.work, estimate.max_size, estimate.output_size) == (162, 81, 81)
    np.testing.assert_array_equal(_contract(cores, candidate, estimate.path), _contract(cores, candidate))


@pytest.mark.parametrize("output", [("b", "a"), ("a",), ()])
def test_single_core_reductions_and_reordering(output):
    cores = {"ab": _core(("a", "b"), (9, 2))}
    cores["ab"].values = np.arange(18).reshape(9, 2)
    candidate = _candidate(cores, output)
    estimate = estimate_candidate_cost(cores, candidate)
    assert estimate.path == ((0,),)
    assert estimate.work == estimate.max_size == 18
    np.testing.assert_array_equal(_contract(cores, candidate, estimate.path), _contract(cores, candidate))


@pytest.mark.parametrize("count", [1, 2, 3])
def test_scalar_inputs_and_outputs(count):
    cores = {str(pos): _core((), (), value=pos + 2) for pos in range(count)}
    candidate = _candidate(cores, ())
    estimate = estimate_candidate_cost(cores, candidate)
    assert estimate.max_size == estimate.output_size == 1
    np.testing.assert_array_equal(_contract(cores, candidate, estimate.path), _contract(cores, candidate))


def test_deterministic_greedy_path_avoids_outer_product():
    cores = {"ab": _core(("a", "b"), (2, 2)), "cd": _core(("c", "d"), (2, 2)),
             "bc": _core(("b", "c"), (2, 2))}
    candidate = _candidate(cores, ("a", "d"))
    first = estimate_candidate_cost(cores, candidate)
    assert first == estimate_candidate_cost(cores, candidate)
    assert first.path[0] == (0, 2)
    assert first.max_size == 4
    np.testing.assert_array_equal(_contract(cores, candidate, first.path), _contract(cores, candidate))


def test_metadata_only_cores_need_no_values_or_allocation():
    cores = {"ab": SimpleNamespace(colors=("a", "b"), shape=(9, 2))}
    assert estimate_candidate_cost(cores, _candidate(cores, ("a",))).output_size == 9


@pytest.mark.parametrize("keys,output", [((), ()), (("ab", "ab"), ("a",)),
                                        (("ab",), ("a", "a")), (("ab",), ("missing",))])
def test_invalid_candidates(keys, output):
    with pytest.raises(ValueError):
        estimate_candidate_cost({"ab": _core(("a", "b"), (2, 2))}, _candidate(keys, output))


def test_missing_core_is_rejected():
    with pytest.raises(KeyError):
        estimate_candidate_cost({}, _candidate(("missing",), ()))


@pytest.mark.parametrize("colors,shape", [(('a', 'a'), (2, 2)), (('a', 'b'), (2,)),
                                         (('a',), (0,)), (('a',), (-1,)),
                                         (('a',), (2.5,)), (('a',), (True,))])
def test_invalid_core_metadata(colors, shape):
    with pytest.raises(ValueError):
        estimate_candidate_cost({"bad": SimpleNamespace(colors=colors, shape=shape)},
                                _candidate(("bad",), ()))


def test_inconsistent_dimensions_are_rejected():
    cores = {"ab": _core(("a", "b"), (2, 2)), "bc": _core(("b", "c"), (9, 2))}
    with pytest.raises(ValueError):
        estimate_candidate_cost(cores, _candidate(cores, ("a",)))
