import numpy as np
import pytest
from tnreason import engine
from demonstrations.sudoku.inferenceClusters.mac import add_sudoku_pair_clusters, prepare_sudoku_state
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.mac import ConstraintNetworkSolver
from experiments.constraint_networks.sudoku_tests.standard_constraints import get_sudoku_constraint_network


def _core(values, colors):
    return engine.get_core("NumpyCore")(values=values, colors=colors)


def _network():
    return ConstraintNetwork({"xy": _core(np.eye(2), ["x", "y"]),
                              "xyz": _core(np.ones((2,) * 3), ["x", "y", "z"])})


def _assert_tensors(state, tensors):
    for color, tensor in tensors.items():
        assert state._domain_tensors[color] is tensor
        assert tuple(np.flatnonzero(tensor)) == state.domains[color]


def test_additions_reuse_original_caches_and_queue_only_new_constraints():
    network = _network()
    state = network.create_mac_state()
    assert state.propagate()
    tensors, caches, originals = dict(state._domain_tensors), dict(state._contractions), dict(state._cores)
    added = _core(np.diag([1, -2]), ["y", "z"])
    assert state.add_cores({"yz": added}) is state
    assert tuple(state._queue) == ("yz",)
    assert network.coresDict.keys() == originals.keys()
    assert state._cores["yz"] is not added
    for key in originals:
        assert state._cores[key] is originals[key] and state._contractions[key] is caches[key]
    added.values.fill(0)
    assert state.propagate().queue_pops == 1
    assert state._contractions["yz"][0].dtype == bool
    _assert_tensors(state, tensors)

    root = state.snapshot_domains()
    for value in (0, 1):
        state.push_level()
        state.assign("x", value)
        assert state.propagate() and state.snapshot_domains() == {color: (value,) for color in root}
        state.push_level()
        state.assign("z", 1 - value)
        assert not state.propagate()
        state.pop_level()
        assert not state.contradiction
        state.pop_level()
        assert state.snapshot_domains() == root and "yz" in state._cores
        assert state.propagate().queue_pops == 0
        _assert_tensors(state, tensors)


def test_new_nary_cache_uses_persistent_domain_tensors():
    state = _network().create_mac_state()
    assert state.propagate()
    tensors = dict(state._domain_tensors)
    state.add_cores({"summary": _core(np.eye(2)[:, :, None] * np.ones(2), ["x", "y", "z"])})
    support, args = state._contractions["summary"]
    assert support.dtype == bool and len(args) == 3
    for idx, arguments in enumerate(args):
        expected = [tensor for axis, tensor in enumerate(tensors.values()) if axis != idx]
        assert all(actual is original for actual, original in zip(arguments[2:-1:2], expected))
    assert state.propagate().queue_pops == 1
    _assert_tensors(state, tensors)


def test_additions_preserve_pending_root_work_and_empty_additions_do_nothing():
    state = _network().create_mac_state()
    pending, statistics = tuple(state._queue), state.statistics
    assert state.add_cores({}) is state
    assert tuple(state._queue) == pending and state.statistics == statistics
    state.add_cores({"z_is_1": _core([0, 1], ["z"])})
    assert tuple(state._queue) == pending + ("z_is_1",)
    assert state.propagate() and state.domains["z"] == (1,)


@pytest.mark.parametrize("kind", ["duplicate_key", "unknown_color", "duplicate_color", "rank", "dimension", "values_shape"])
def test_additions_validate_whole_batch_before_mutating_state(kind):
    state = _network().create_mac_state()
    assert state.propagate()
    bad = _core([1, 1], ["x"])
    key = "bad"
    if kind == "duplicate_key":
        key = "xy"
    elif kind == "unknown_color":
        bad.colors = ["unknown"]
    elif kind == "duplicate_color":
        bad = _core(np.ones((2, 2)), ["x", "y"])
        bad.colors = ["x", "x"]
    elif kind == "rank":
        bad.colors = ["x", "y"]
    elif kind == "dimension":
        bad = _core(np.ones(3), ["x"])
    else:
        bad.values = np.ones((2, 2))
    cores, caches, incidence = dict(state._cores), dict(state._contractions), dict(state._core_keys_by_color)
    domains, statistics = state.snapshot_domains(), state.statistics
    with pytest.raises(ValueError):
        state.add_cores({"valid": _core([0, 1], ["x"]), key: bad})
    assert state._cores == cores and state._contractions == caches and state._core_keys_by_color == incidence
    assert state.snapshot_domains() == domains and state.statistics == statistics and not state._queue


@pytest.mark.parametrize("value", [0, 1])
def test_added_scalar_support_is_handled_without_new_variables(value):
    state = _network().create_mac_state()
    assert state.propagate()
    state.add_cores({"scalar": _core(value, [])})
    assert state.contradiction == (value == 0)
    assert bool(state.propagate()) == (value != 0)
    if not value:
        assert "scalar" in state.contradiction_reason


def test_solve_reuses_root_fixed_point_and_never_recreates_state(monkeypatch):
    network = _network()
    state = network.create_mac_state({"x": (1,), "z": (0,)})
    assert state.propagate()
    root, pops, tensors = state.snapshot_domains(), state.statistics.queue_pops, dict(state._domain_tensors)
    monkeypatch.setattr(network, "create_mac_state", lambda *args, **kwargs: pytest.fail("Repeated state preparation"))
    solver = ConstraintNetworkSolver(network)
    for solve in (lambda: network.solve_mac(state=state), lambda: solver.solve(state=state)):
        result = solve()
        assert result.solved and result.assignment == {"x": 1, "y": 1, "z": 0} and result.nodes == 1
        assert result.statistics.queue_pops == pops and state.snapshot_domains() == root
        _assert_tensors(state, tensors)


def test_reused_solve_honors_additions_and_restores_prepared_root():
    network = _network()
    state = network.create_mac_state({"z": (1,)})
    assert state.propagate()
    state.add_cores({"yz": _core(np.eye(2), ["y", "z"])})
    result = network.solve_mac(state=state)
    expected = ConstraintNetwork({**network.coresDict, "yz": _core(np.eye(2), ["y", "z"])}).solve_mac({"z": (1,)})
    assert result.solved and result.assignment == expected.assignment == {"x": 1, "y": 1, "z": 1}
    assert state.decision_level == 0 and not state._queue


def test_reused_search_restores_root_after_solution_and_node_limit():
    network = _network()
    state = network.create_mac_state()
    assert state.propagate()
    state.add_cores({"yz": _core(np.eye(2), ["y", "z"])})
    assert state.propagate()
    root, tensors = state.snapshot_domains(), dict(state._domain_tensors)
    limited = network.solve_mac(max_nodes=1, state=state)
    assert limited.node_limit_reached and not limited.solved and state.snapshot_domains() == root
    solved = network.solve_mac(state=state)
    assert solved.solved and solved.assignment == {"x": 0, "y": 0, "z": 0}
    assert state.snapshot_domains() == root and state.decision_level == 0 and not state._queue
    _assert_tensors(state, tensors)


def test_reuse_rejects_foreign_owner_domains_and_nonroot_states():
    network, foreign = _network(), _network()
    state = network.create_mac_state()
    with pytest.raises(ValueError):
        foreign.solve_mac(state=state)
    with pytest.raises(ValueError):
        network.solve_mac(domains={}, state=state)
    state.push_level()
    with pytest.raises(RuntimeError):
        state.add_cores({"z_is_1": _core([0, 1], ["z"])})
    with pytest.raises(RuntimeError):
        network.solve_mac(state=state)
    assert "z_is_1" not in state._cores and state.decision_level == 1
    state.pop_level()


def test_reversible_branch_additions_restore_cores_caches_incidence_and_tensors():
    network = _network()
    state = network.create_mac_state()
    assert state.propagate()
    root, cores = state.snapshot_domains(), dict(state._cores)
    caches, incidence, tensors = dict(state._contractions), dict(state._core_keys_by_color), dict(state._domain_tensors)
    state.push_level()
    state.assign("x", 1)
    state.add_cores({"yz": _core(np.eye(2), ["y", "z"])}, reversible=True)
    assert state.propagate() and state.snapshot_domains() == {"x": (1,), "y": (1,), "z": (1,)}
    branch, summary, contraction = state.snapshot_domains(), state._cores["yz"], state._contractions["yz"]
    state.push_level()
    state.add_cores({"impossible": _core(0, [])}, reversible=True)
    assert not state.propagate() and "impossible" in state._cores
    state.pop_level()
    assert not state.contradiction and state.snapshot_domains() == branch
    assert "impossible" not in state._cores and "impossible" not in state._contractions
    assert state._cores["yz"] is summary and state._contractions["yz"] is contraction
    state.pop_level()
    assert state.snapshot_domains() == root and state._cores == cores
    assert state._contractions == caches and state._core_keys_by_color == incidence
    assert state._added_core_keys == {"yz", "impossible"}
    assert not state._queue and network.coresDict.keys() == cores.keys()
    assert all(state._cores[key] is core and state._contractions[key] is caches[key] for key, core in cores.items())
    _assert_tensors(state, tensors)

    state.push_level()
    state.add_cores({"yz": _core(np.eye(2), ["y", "z"])}, reversible=True)
    state.assign("x", 0)
    assert state.propagate() and state.snapshot_domains() == {"x": (0,), "y": (0,), "z": (0,)}
    state.pop_level()
    assert state.snapshot_domains() == root and state._core_keys_by_color == incidence


def test_reversible_additions_restore_pending_queue_and_validate_batches_atomically():
    state = _network().create_mac_state()
    pending, cores = tuple(state._queue), dict(state._cores)
    caches, incidence = dict(state._contractions), dict(state._core_keys_by_color)
    state.push_level()
    with pytest.raises(ValueError):
        state.add_cores({"valid": _core([0, 1], ["x"]), "xy": _core(np.eye(2), ["x", "y"])}, reversible=True)
    assert state._cores == cores and state._contractions == caches and state._core_keys_by_color == incidence
    assert tuple(state._queue) == pending
    state.add_cores({"impossible": _core(0, [])}, reversible=True)
    assert not state.propagate()
    state.pop_level()
    assert not state.contradiction and tuple(state._queue) == pending
    assert state._cores == cores and state._contractions == caches and state._core_keys_by_color == incidence
    assert state.propagate()


def test_optional_closure_hook_receives_fixed_point_changes_and_reaches_combined_fixed_point():
    network = _network()
    state = network.create_mac_state()
    closures = []

    def on_closure(current, changed):
        assert not current._queue and current.domains["x"] == current.domains["y"]
        closures.append((frozenset(changed), current.snapshot_domains()))
        if current.domains["x"] == (1,) and "yz" not in current._cores:
            return {"yz": _core(np.eye(2), ["y", "z"])}

    assert state.on_closure is None
    state.on_closure = on_closure
    assert state.propagate() and closures[0][0] == frozenset(state.dimensions)
    state.push_level()
    state.assign("x", 1)
    result = state.propagate()
    assert result and state.is_solved() and state.domains["z"] == (1,)
    assert closures[-2][0] == frozenset(("x", "y")) and closures[-1][0] == frozenset(("z",))
    assert result.domain_reductions == 2 and "yz" in state._cores
    state.pop_level()
    assert "yz" not in state._cores and state.snapshot_domains() == {color: (0, 1) for color in state.dimensions}
    assert state.propagate().queue_pops == 0
    assert closures[-1][0] == frozenset(("x", "y", "z"))


def test_closure_hook_is_skipped_on_contradiction_and_default_mac_is_unchanged():
    network = _network()
    ordinary, hooked = network.create_mac_state(), network.create_mac_state()
    calls = []
    hooked.on_closure = lambda current, changed: calls.append(current.snapshot_domains())
    assert ordinary.propagate() == hooked.propagate() and calls == [ordinary.snapshot_domains()]
    assert ordinary.on_closure is None
    for state in (ordinary, hooked):
        state.push_level()
        state.assign("x", 0)
        state.assign("y", 1)
        assert not state.propagate()
        state.pop_level()
    assert len(calls) == 1 and ordinary.snapshot_domains() == hooked.snapshot_domains()
    assert network.solve_mac(state=ordinary).assignment == network.solve_mac(state=hooked).assignment


def test_search_restores_dynamic_closure_additions_after_solution_and_node_limit():
    network = _network()
    state = network.create_mac_state()
    state.on_closure = lambda current, changed: ({"yz": _core(np.eye(2), ["y", "z"])}
        if len(current.domains["x"]) == 1 and "yz" not in current._cores else None)
    assert state.propagate()
    root, cores, incidence = state.snapshot_domains(), dict(state._cores), dict(state._core_keys_by_color)
    limited = network.solve_mac(state=state, max_nodes=1)
    assert limited.node_limit_reached and not limited.solved
    assert state.snapshot_domains() == root and state._cores == cores and state._core_keys_by_color == incidence
    solved = network.solve_mac(state=state)
    assert solved.solved and solved.assignment == {"x": 0, "y": 0, "z": 0} and solved.nodes == 2
    assert state.snapshot_domains() == root and state._cores == cores and state._core_keys_by_color == incidence
    assert state.decision_level == 0 and not state._queue


def test_reused_contradictory_root_remains_unsatisfiable():
    network = _network()
    state = network.create_mac_state({"x": ()})
    assert not state.propagate()
    result = network.solve_mac(state=state)
    assert result.unsatisfiable and result.root_contradiction and not result.solved


def test_sudoku_preparation_creates_only_one_state_and_reuses_original_tensors(monkeypatch):
    network = ConstraintNetwork(get_sudoku_constraint_network(num=2))
    domains = {"pos_0_0_0_0": (0, 1), "pos_0_0_1_0": (0, 1)}
    create, calls = network.create_mac_state, []

    def capture(*args, **kwargs):
        state = create(*args, **kwargs)
        calls.append((state, dict(state._domain_tensors), dict(state._contractions)))
        return state

    monkeypatch.setattr(network, "create_mac_state", capture)
    state = prepare_sudoku_state(network, domains=domains, num=2)
    assert len(calls) == 1 and state is calls[0][0]
    tensors = calls[0][1]
    summaries = state._cores.keys() - network.coresDict.keys()
    assert summaries and set(state._queue) == summaries
    assert all(state._contractions[key] is cache for key, cache in calls[0][2].items())
    _assert_tensors(state, calls[0][1])
    expected = add_sudoku_pair_clusters(network, domains=domains, num=2).solve_mac(domains=domains)
    calls.clear()
    result = network.solve_mac(state=state)
    assert not calls and result.solved and result.assignment == expected.assignment and result.nodes == expected.nodes
    assert "pos_0_0_0_0" in state.domains and state.decision_level == 0
    _assert_tensors(state, tensors)


@pytest.mark.parametrize("domains", [None, {"pos_0_0_0_0": ()}])
def test_sudoku_preparation_without_selected_pairs_never_requeues_originals(domains):
    network = ConstraintNetwork(get_sudoku_constraint_network(num=2))
    state = prepare_sudoku_state(network, domains=domains, num=2)
    assert state._cores.keys() == network.coresDict.keys()
    pops = state.statistics.queue_pops
    if domains is None:
        assert not state._queue and state.propagate().queue_pops == 0
    else:
        result = network.solve_mac(state=state)
        assert result.unsatisfiable and result.root_contradiction
        assert state.statistics.queue_pops == pops
