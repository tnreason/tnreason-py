from itertools import product
import numpy as np
import pytest
from tnreason import engine
from demonstrations.sudoku.inferenceClusters.mac import prepare_sudoku_state
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.sudoku_tests.standard_constraints import get_sudoku_constraint_network


def _network(num=2):
    return ConstraintNetwork(get_sudoku_constraint_network(num=num))


def _prepare(network, domains=None, num=2):
    return prepare_sudoku_state(network, domains=domains, num=num, pairs=False, locked=True)


def _summaries(network, state):
    return {key: core for key, core in state._cores.items() if key not in network.coresDict}


def _summary(network, state, kind="row"):
    matching = [core for core in _summaries(network, state).values()
                if set(core.colors) == {f"{kind}_0_0_0", "square_0_0_0"}]
    assert len(matching) == 1
    return matching[0]


def _pattern(kind, pointing):
    unit, box = f"{kind}_0_0_0", "square_0_0_0"
    box_domain = (0, 1) if kind == "row" else (0, 2)
    return (box, unit, box_domain, (0, 1)) if pointing else (unit, box, (0, 1), box_domain)


@pytest.mark.parametrize("kind", ["row", "col"])
@pytest.mark.parametrize("pointing", [True, False], ids=["pointing", "claiming"])
def test_locked_summaries_strengthen_gac_in_both_directions(kind, pointing):
    network = _network()
    source, target, selected, expected = _pattern(kind, pointing)
    ordinary = network.create_mac_state({source: selected})
    assert ordinary.propagate() and ordinary.domains[target] == (0, 1, 2, 3)
    state = _prepare(network, {source: selected})
    assert state.propagate() and state.domains[target] == expected
    assert all(len(core.colors) == 2 for core in _summaries(network, state).values())


@pytest.mark.parametrize("kind", ["row", "col"])
@pytest.mark.parametrize("pointing", [True, False], ids=["pointing", "claiming"])
def test_standard_nine_by_nine_activates_small_boolean_locked_core(kind, pointing):
    network = _network(3)
    unit, box = f"{kind}_0_0_0", "square_0_0_0"
    box_domain, unit_domain = ((0, 1, 2) if kind == "row" else (0, 3, 6)), (0, 1, 2)
    source, target, selected, expected = ((box, unit, box_domain, unit_domain) if pointing
                                         else (unit, box, unit_domain, box_domain))
    state = _prepare(network, {source: selected}, num=3)
    assert state.domains[target] == expected and not state._queue
    summary = _summary(network, state, kind)
    assert summary.values.dtype == bool and summary.values.shape == (9, 9)
    assert np.count_nonzero(summary.values) == 39


@pytest.mark.parametrize("kind", ["row", "col"])
@pytest.mark.parametrize("pointing", [True, False], ids=["pointing", "claiming"])
def test_later_branch_patterns_activate_cached_summaries_and_restore_siblings(kind, pointing, monkeypatch):
    network = _network()
    state = _prepare(network)
    assert not _summaries(network, state) and state.propagate().queue_pops == 0
    root, tensors = state.snapshot_domains(), dict(state._domain_tensors)
    caches, incidence = dict(state._contractions), dict(state._core_keys_by_color)
    source, target, selected, expected = _pattern(kind, pointing)
    contract, calls = engine.contract, []

    def capture(*args, **kwargs):
        calls.append(kwargs["openColors"])
        return contract(*args, **kwargs)

    monkeypatch.setattr(engine, "contract", capture)

    for allowed, result in ((selected, expected), (tuple(value for value in range(4) if value not in selected),
                                                tuple(value for value in range(4) if value not in expected))):
        state.push_level()
        state.restrict_domain(source, allowed)
        assert state.propagate() and state.domains[target] == result
        summaries, branch = _summaries(network, state), state.snapshot_domains()
        assert summaries
        state.push_level()
        state.assign(target, next(value for value in range(4) if value not in result))
        assert not state.propagate()
        state.pop_level()
        assert state.snapshot_domains() == branch and not state.contradiction
        assert all(state._cores[key] is core for key, core in summaries.items())
        state.pop_level()
        assert state.snapshot_domains() == root and state.propagate().queue_pops == 0
        assert not _summaries(network, state) and state._core_keys_by_color == incidence
        for color, tensor in tensors.items():
            assert state._domain_tensors[color] is tensor and tuple(np.flatnonzero(tensor)) == root[color]
        assert all(state._contractions[key] is cached for key, cached in caches.items())

    assert calls
    monkeypatch.setattr(engine, "contract", lambda *args, **kwargs: pytest.fail("Repeated branch rebuilt a summary"))
    state.push_level()
    state.restrict_domain(source, selected)
    assert state.propagate() and state.domains[target] == expected
    state.pop_level()
    assert state.snapshot_domains() == root and not _summaries(network, state)


@pytest.mark.parametrize("num", [2, 3])
def test_no_locked_pattern_adds_no_cores_or_contractions(num, monkeypatch):
    network = _network(num)
    original, values = dict(network.coresDict), {key: core.values.copy() for key, core in network.coresDict.items()}
    monkeypatch.setattr(engine, "contract", lambda *args, **kwargs: pytest.fail("Contracted an inactive locked motif"))
    state = _prepare(network, num=num)
    assert not _summaries(network, state) and not state._queue
    for key, core in original.items():
        assert network.coresDict[key] is core
        np.testing.assert_array_equal(core.values, values[key])


@pytest.mark.parametrize("kind", ["row", "col"])
@pytest.mark.parametrize("pointing", [True, False], ids=["pointing", "claiming"])
def test_already_pruned_locked_target_does_not_add_redundant_core(kind, pointing):
    network = _network()
    source, target, selected, expected = _pattern(kind, pointing)
    state = _prepare(network, {source: selected, target: expected})
    assert not any(set(core.colors) == {source, target} for core in _summaries(network, state).values())


def test_locked_activation_repeats_after_cascaded_gac_until_fixed_point(monkeypatch):
    bridge = engine.get_core("NumpyCore")(values=(np.arange(4)[:, None] < 2) == (np.arange(4)[None, :] < 2),
                                           colors=["row_0_0_0", "square_1_1_0"])
    network = ConstraintNetwork({**_network().coresDict, "cascade_bridge": bridge})
    ordinary = network.create_mac_state({"square_0_0_0": (0, 1)})
    assert ordinary.propagate() and ordinary.domains["square_0_0_1"] == (0, 1, 2, 3)
    assert ordinary.domains["square_1_1_0"] == ordinary.domains["row_1_1_0"] == (0, 1, 2, 3)
    state = _prepare(network, {"square_0_0_0": (0, 1)})
    assert state.domains["row_0_0_0"] == (0, 1)
    assert state.domains["row_0_0_1"] == state.domains["square_0_0_1"] == (2, 3)
    assert state.domains["square_1_1_0"] == state.domains["row_1_1_0"] == (0, 1)
    assert not state._queue and not state.contradiction
    summaries, pops = _summaries(network, state), state.statistics.queue_pops
    assert len(summaries) >= 3
    monkeypatch.setattr(engine, "contract", lambda *args, **kwargs: pytest.fail("Fixed point contracted again"))
    assert state.propagate().queue_pops == 0 and state.statistics.queue_pops == pops
    assert all(state._cores[key] is core for key, core in summaries.items())


@pytest.mark.parametrize("kind", ["row", "col"])
@pytest.mark.parametrize("changed", [False, True], ids=["standard", "changed_support"])
def test_locked_summary_is_exact_projection_and_contains_no_selection_domains(kind, changed):
    network = _network()
    if changed:
        core = network.coresDict[f"{kind}_0_0_0_a_0_0_0_0_0"]
        core.values[tuple(3 if color == f"{kind}_0_0_0" else 0 for color in core.colors)] = 0
    source, _, selected, _ = _pattern(kind, True)
    summary = _summary(network, _prepare(network, {source: selected}), kind)
    atoms = ([f"a_0_0_0_{idx}_0" for idx in range(2)] if kind == "row"
             else [f"a_0_{idx}_0_0_0" for idx in range(2)])
    links = [core for core in network.coresDict.values()
             if len(core.colors) == 2 and set(core.colors).issubset(set(summary.colors) | set(atoms))]
    assert len(links) == 4
    for placements in product(range(4), repeat=2):
        evidence = dict(zip(summary.colors, placements))
        supported = False
        for values in product(range(2), repeat=2):
            evidence.update(zip(atoms, values))
            if all(core[{color: evidence[color] for color in core.colors}] != 0 for core in links):
                supported = True
                break
        assert bool(summary.values[placements]) == supported
    other_source, _, other_selected, _ = _pattern(kind, False)
    np.testing.assert_array_equal(summary.values, _summary(network, _prepare(network, {other_source: other_selected}), kind).values)
    assert summary.values.dtype == bool and summary.values.shape == (4, 4)
    if not changed:
        assert np.count_nonzero(summary.values) == 6


def test_locked_contractions_use_only_original_intersection_links(monkeypatch):
    network = _network()
    contract, calls = engine.contract, []

    def capture(coreDict, openColors, **kwargs):
        assert len(coreDict) == 4 and set(coreDict).issubset(network.coresDict)
        assert len(openColors) == 2 and any(color.startswith("square_") for color in openColors)
        assert all(len(core.colors) == 2 and core.values.dtype == bool for core in coreDict.values())
        calls.append((coreDict, openColors))
        return contract(coreDict, openColors, **kwargs)

    monkeypatch.setattr(engine, "contract", capture)
    state = _prepare(network, {"square_0_0_0": (0, 1)})
    assert calls and 0 < len(_summaries(network, state)) < 64


def test_signed_weights_preserve_all_locked_supports():
    positive, signed = _network(), _network()
    for core in signed.coresDict.values():
        core.values *= np.where(np.indices(core.shape).sum(axis=0) % 2, -2, 3)
    domains = {"square_0_0_0": (0, 1)}
    reference, actual = _summaries(positive, _prepare(positive, domains)), _summaries(signed, _prepare(signed, domains))
    assert reference
    assert actual.keys() == reference.keys()
    for key, core in actual.items():
        assert core.colors == reference[key].colors
        np.testing.assert_array_equal(core.values, reference[key].values)


def test_sparse_overlapping_slices_use_realized_support_without_mutating_source():
    network = _network()
    domains = {"square_0_0_0": (0, 1)}
    reference = _summary(network, _prepare(network, domains))
    key = "row_0_0_0_a_0_0_0_0_0"
    original = network.coresDict[key]
    cancelled = {"a_0_0_0_0_0": 0, "row_0_0_0": 0}
    slices = [(value, assignment) for value, assignment in original if value != 0]
    slices += [(5, cancelled), (-5, cancelled)]
    sparse = engine.create_from_slice_iterator(original.shape, original.colors, slices, coreType="PolynomialCore")
    network.coresDict[key] = sparse
    actual = _summary(network, _prepare(network, domains))
    assert sparse[cancelled] == 0 and sparse.values == slices and network.coresDict[key] is sparse
    np.testing.assert_array_equal(actual.values, reference.values)


@pytest.mark.parametrize("kind", ["row", "col", "square"])
def test_locked_preparation_rejects_missing_intersection_links(kind):
    network = _network()
    del network.coresDict[f"{kind}_0_0_0_a_0_0_0_0_0"]
    with pytest.raises(ValueError, match="Missing Sudoku link"):
        _prepare(network)


def test_locked_rules_are_opt_in_and_can_be_used_without_pair_rules(monkeypatch):
    network = _network()
    state = prepare_sudoku_state(network, num=2)
    assert not _summaries(network, state)
    monkeypatch.setattr("demonstrations.sudoku.inferenceClusters.mac.detect_naked_pairs",
                        lambda *args, **kwargs: pytest.fail("Pair selection was enabled"))
    monkeypatch.setattr("demonstrations.sudoku.inferenceClusters.mac.detect_hidden_pairs",
                        lambda *args, **kwargs: pytest.fail("Pair selection was enabled"))
    assert _summaries(network, _prepare(network, {"square_0_0_0": (0, 1)}))


@pytest.mark.parametrize("contradiction", [False, True])
def test_solved_and_contradictory_roots_skip_unneeded_locked_preparation(contradiction):
    network = _network()
    grid = ((0, 1, 2, 3), (2, 3, 0, 1), (1, 0, 3, 2), (3, 2, 1, 0))
    domains = ({"pos_0_0_0_0": ()} if contradiction else {
        f"pos_{row // 2}_{row % 2}_{col // 2}_{col % 2}": (grid[row][col],)
        for row in range(4) for col in range(4)})
    state = _prepare(network, domains)
    assert not _summaries(network, state)
    assert state.contradiction if contradiction else state.is_solved()


def test_pair_and_locked_summaries_combine_and_solution_satisfies_all_cores():
    network = _network()
    state = prepare_sudoku_state(network, {"pos_0_0_0_0": (0, 1), "pos_0_0_1_0": (0, 1)},
                                    num=2, locked=True)
    summaries = _summaries(network, state)
    assert any(len(core.colors) == 4 for core in summaries.values())
    assert all(len(core.colors) in (2, 4) for core in summaries.values()) and not state._queue
    result = network.solve_mac(state=state)
    assert result.solved and result.assignment is not None
    for core in state._cores.values():
        assert core[{color: result.assignment[color] for color in core.colors}] != 0
