from itertools import product
import numpy as np
import pytest
from tnreason import engine
from demonstrations.sudoku.inferenceClusters.mac import add_sudoku_pair_clusters
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.sudoku_tests.standard_constraints import get_sudoku_constraint_network


def _cell(row, col):
    return row // 2, row % 2, col // 2, col % 2


def _pos(cell):
    return "pos_" + "_".join(map(str, cell))


def _unit(kind):
    if kind == "row":
        return [_cell(0, col) for col in range(4)], (0, 2)
    if kind == "col":
        return [_cell(row, 0) for row in range(4)], (0, 2)
    return [_cell(row, col) for row in range(2) for col in range(2)], (0, 3)


def _domains(kind="row", hidden=False):
    cells, pair = _unit(kind)
    if hidden:
        return {_pos(cell): (0, 1, 2) if idx in pair else (2, 3)
                for idx, cell in enumerate(cells)}
    return {_pos(cells[idx]): (0, 1) for idx in pair}


def _summary(original, augmented, kind="row"):
    cells, pair = _unit(kind)
    colors = {_pos(cells[idx]) for idx in pair} | {f"{kind}_{digit}_0_0" for digit in (0, 1)}
    matching = [core for key, core in augmented.coresDict.items()
                if key not in original.coresDict and set(core.colors) == colors]
    assert len(matching) == 1
    return matching[0]


@pytest.mark.parametrize("kind", ["row", "col", "square"])
@pytest.mark.parametrize("hidden", [False, True], ids=["naked", "hidden"])
def test_optional_pair_contractions_strengthen_original_gac(kind, hidden):
    network = ConstraintNetwork(get_sudoku_constraint_network(num=2))
    domains = _domains(kind, hidden)
    ordinary = network.create_mac_state(domains)
    assert ordinary.propagate()
    cells, pair = _unit(kind)
    for idx in pair:
        assert ordinary.domains[_pos(cells[idx])] == ((0, 1, 2) if hidden else (0, 1))
    if not hidden:
        assert ordinary.domains[f"{kind}_0_0_0"] == (0, 1, 2, 3)

    augmented = add_sudoku_pair_clusters(network, domains=domains, num=2)
    _summary(network, augmented, kind)
    state = augmented.create_mac_state(domains)
    assert state.propagate()
    for idx in pair:
        assert state.domains[_pos(cells[idx])] == (0, 1)
    for idx, cell in enumerate(cells):
        if idx not in pair:
            assert state.domains[_pos(cell)] == (2, 3)
    for digit in (0, 1):
        assert state.domains[f"{kind}_{digit}_0_0"] == pair


@pytest.mark.parametrize("maintain_indices", [False, True])
def test_preprocessing_preserves_original_cores_and_adds_boolean_tensors(maintain_indices):
    network = ConstraintNetwork(get_sudoku_constraint_network(num=2), maintain_indices=maintain_indices)
    cores, values = dict(network.coresDict), {key: core.values.copy() for key, core in network.coresDict.items()}
    augmented = add_sudoku_pair_clusters(network, domains=_domains(), num=2)
    assert augmented is not network
    assert network.coresDict == cores
    for key, core in cores.items():
        assert augmented.coresDict[key] is core
        np.testing.assert_array_equal(core.values, values[key])
    summaries = [core for key, core in augmented.coresDict.items() if key not in cores]
    assert summaries
    assert all(core.values.dtype == bool and core.values.shape == (4,) * 4 for core in summaries)


def test_pair_selected_by_both_detectors_is_contracted_only_once():
    network = ConstraintNetwork(get_sudoku_constraint_network(num=2))
    cells, pair = _unit("row")
    domains = {_pos(cell): (0, 1) if idx in pair else (2, 3) for idx, cell in enumerate(cells)}
    augmented = add_sudoku_pair_clusters(network, domains=domains, num=2)
    _summary(network, augmented)  # Exactly one summary for this output-color set.


def test_summaries_are_contracted_from_only_eight_existing_links(monkeypatch):
    network = ConstraintNetwork(get_sudoku_constraint_network(num=2))
    contract, calls = engine.contract, []

    def capture(coreDict, openColors, **kwargs):
        assert len(coreDict) == 8
        assert set(coreDict).issubset(network.coresDict)
        assert len(openColors) == 4 and all(not color.startswith("a_") for color in openColors)
        assert all(len(core.colors) == 2 for core in coreDict.values())
        calls.append((coreDict, openColors))
        return contract(coreDict, openColors, **kwargs)

    monkeypatch.setattr(engine, "contract", capture)
    augmented = add_sudoku_pair_clusters(network, domains=_domains(), num=2)
    assert calls and len(calls) == len(augmented.coresDict) - len(network.coresDict)


def test_summary_matches_all_projected_assignments_of_its_original_links():
    network = ConstraintNetwork(get_sudoku_constraint_network(num=2))
    augmented = add_sudoku_pair_clusters(network, domains=_domains(), num=2)
    summary = _summary(network, augmented)
    cells, pair = _unit("row")
    atoms = ["a_" + "_".join(map(str, (*cells[idx], digit))) for idx in pair for digit in (0, 1)]
    links = [core for core in network.coresDict.values()
             if set(core.colors).issubset(set(summary.colors) | set(atoms))]
    assert len(links) == 8
    for values in product(range(4), repeat=4):
        evidence = dict(zip(summary.colors, values))
        supported = False
        for atom_values in product(range(2), repeat=4):
            evidence.update(zip(atoms, atom_values))
            if all(core[{color: evidence[color] for color in core.colors}] != 0 for core in links):
                supported = True
                break
        assert bool(summary.values[values]) == supported


def test_signed_core_weights_do_not_change_contracted_support():
    positive = ConstraintNetwork(get_sudoku_constraint_network(num=2))
    signed = ConstraintNetwork(get_sudoku_constraint_network(num=2))
    for core in signed.coresDict.values():
        core.values *= np.where(np.indices(core.shape).sum(axis=0) % 2, -2, 3)
    positive_augmented = add_sudoku_pair_clusters(positive, domains=_domains(), num=2)
    signed_augmented = add_sudoku_pair_clusters(signed, domains=_domains(), num=2)
    reference = _summary(positive, positive_augmented)
    actual = _summary(signed, signed_augmented)
    assert actual.colors == reference.colors
    np.testing.assert_array_equal(actual.values, reference.values)


def test_selected_summaries_do_not_embed_selection_domains():
    network = ConstraintNetwork(get_sudoku_constraint_network(num=2))
    augmented = add_sudoku_pair_clusters(network, domains=_domains(), num=2)
    unrestricted = augmented.create_mac_state()
    assert unrestricted.propagate()
    assert all(len(domain) == unrestricted.dimensions[color] for color, domain in unrestricted.domains.items())
    grid = ((2, 1, 0, 3), (0, 3, 2, 1), (1, 2, 3, 0), (3, 0, 1, 2))
    witness = {_pos(_cell(row, col)): (grid[row][col],) for row in range(4) for col in range(4)}
    state = augmented.create_mac_state(witness)
    assert state.propagate() and state.is_solved()
    assignment = state.assignment()
    assert all(core[{color: assignment[color] for color in core.colors}] != 0
               for core in augmented.coresDict.values())


def test_cluster_domains_and_tensors_restore_in_nested_and_sibling_branches():
    network = ConstraintNetwork(get_sudoku_constraint_network(num=2))
    augmented = add_sudoku_pair_clusters(network, domains=_domains(), num=2)
    state = augmented.create_mac_state(_domains())
    assert state.propagate()
    root, tensors = state.snapshot_domains(), dict(state._domain_tensors)
    cells, pair = _unit("row")
    first, second = (_pos(cells[idx]) for idx in pair)
    for value in (0, 1):
        state.push_level()
        state.assign(first, value)
        assert state.propagate()
        assert state.domains[second] == (1 - value,)
        branch = state.snapshot_domains()
        state.push_level()
        state.assign(second, value)
        assert not state.propagate()
        state.pop_level()
        assert not state.contradiction and state.snapshot_domains() == branch
        state.pop_level()
        assert state.snapshot_domains() == root and state.propagate().queue_pops == 0
        for color, tensor in tensors.items():
            assert state._domain_tensors[color] is tensor
            assert tuple(np.flatnonzero(tensor)) == root[color]


def test_no_pairs_returns_an_unmodified_copy():
    network = ConstraintNetwork(get_sudoku_constraint_network(num=2))
    augmented = add_sudoku_pair_clusters(network, num=2)
    assert augmented is not network and augmented.coresDict == network.coresDict


def test_sudoku_helper_rejects_non_sudoku_networks():
    core = engine.get_core()(values=np.eye(2), colors=["x", "y"])
    with pytest.raises(ValueError):
        add_sudoku_pair_clusters(ConstraintNetwork({"xy": core}), num=2)


def test_sudoku_helper_rejects_missing_categorical_variables():
    cores = {key: core for key, core in get_sudoku_constraint_network(num=2).items()
             if "row_0_0_0" not in core.colors}
    with pytest.raises(ValueError):
        add_sudoku_pair_clusters(ConstraintNetwork(cores), num=2)


def test_sudoku_helper_rejects_wrong_categorical_dimensions():
    cores = get_sudoku_constraint_network(num=2)
    for core in cores.values():
        if "row_0_0_0" in core.colors:
            core.values = np.zeros((2, 5))
            core.shape = core.values.shape
    with pytest.raises(ValueError):
        add_sudoku_pair_clusters(ConstraintNetwork(cores), num=2)


def test_sudoku_helper_rejects_non_boolean_indicator_dimensions():
    cores = get_sudoku_constraint_network(num=2)
    for core in cores.values():
        if "a_0_0_0_0_0" in core.colors:
            core.values = np.zeros((3, 4))
            core.shape = core.values.shape
    with pytest.raises(ValueError):
        add_sudoku_pair_clusters(ConstraintNetwork(cores), num=2)


def test_sudoku_helper_rejects_missing_links_of_selected_pair():
    cores = get_sudoku_constraint_network(num=2)
    del cores["pos_0_0_0_0_a_0_0_0_0_0"]
    with pytest.raises(ValueError):
        add_sudoku_pair_clusters(ConstraintNetwork(cores), domains=_domains(), num=2)


def test_repeated_preprocessing_keeps_existing_summaries():
    network = ConstraintNetwork(get_sudoku_constraint_network(num=2))
    augmented = add_sudoku_pair_clusters(network, domains=_domains(), num=2)
    repeated = add_sudoku_pair_clusters(augmented, domains=_domains(), num=2)
    assert all(repeated.coresDict[key] is core for key, core in augmented.coresDict.items())
    _summary(network, repeated)


def test_existing_summary_name_does_not_overwrite_user_core():
    network = ConstraintNetwork(get_sudoku_constraint_network(num=2))
    augmented = add_sudoku_pair_clusters(network, domains=_domains(), num=2)
    summary = _summary(network, augmented)
    key = next(key for key, core in augmented.coresDict.items() if core is summary)
    user_core = engine.get_core()(values=np.ones((4,) * 4), colors=summary.colors)
    network.coresDict[key] = user_core
    repeated = add_sudoku_pair_clusters(network, domains=_domains(), num=2)
    assert repeated.coresDict[key] is network.coresDict[key] is user_core
    np.testing.assert_array_equal(user_core.values, np.ones((4,) * 4))


def test_contradictory_selection_domains_return_unchanged_copy():
    network = ConstraintNetwork(get_sudoku_constraint_network(num=2))
    augmented = add_sudoku_pair_clusters(network, domains={"pos_0_0_0_0": ()}, num=2)
    assert augmented is not network and augmented.coresDict == network.coresDict


def test_sparse_overlapping_slices_use_realized_nonzero_support():
    network = ConstraintNetwork(get_sudoku_constraint_network(num=2))
    reference = _summary(network, add_sudoku_pair_clusters(network, domains=_domains(), num=2))
    key = "pos_0_0_0_0_a_0_0_0_0_0"
    original = network.coresDict[key]
    cancelled = {"a_0_0_0_0_0": 0, "pos_0_0_0_0": 0}
    slices = [(-2 * value, assignment) for value, assignment in original if value != 0]
    slices += [(5, cancelled), (-5, cancelled)]
    polynomial = engine.create_from_slice_iterator(original.shape, original.colors, slices, coreType="PolynomialCore")
    network.coresDict[key] = polynomial
    actual = _summary(network, add_sudoku_pair_clusters(network, domains=_domains(), num=2))
    assert polynomial[cancelled] == 0 and polynomial.values == slices
    np.testing.assert_array_equal(actual.values, reference.values)


def test_partially_consumed_numpy_iterator_does_not_truncate_support():
    reference_network = ConstraintNetwork(get_sudoku_constraint_network(num=2))
    reference = _summary(reference_network, add_sudoku_pair_clusters(reference_network, domains=_domains(), num=2))
    network = ConstraintNetwork(get_sudoku_constraint_network(num=2))
    core = network.coresDict["pos_0_0_0_0_a_0_0_0_0_0"]
    for _ in range(3):
        next(core)
    assert core.index == 3
    actual = _summary(network, add_sudoku_pair_clusters(network, domains=_domains(), num=2))
    np.testing.assert_array_equal(actual.values, reference.values)
    assert core.index == 3
