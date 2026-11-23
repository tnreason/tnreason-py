"""The explicit experiment must preserve generic MAC and sibling domains."""
from demonstrations.sudoku.inferenceClusters import RuleLimits
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.sudoku_tests.standard_constraints import get_sudoku_constraint_network
from experiments.sudoku_rule_clusters import ClusterClosure


def test_exact_cluster_projection_adds_information_without_adding_cores():
    network = ConstraintNetwork(get_sudoku_constraint_network(num=3), maintain_indices=True)
    domains = {"square_0_0_0": (0, 1, 2)}
    ordinary = network.create_mac_state(domains)
    assert ordinary.on_closure is None and ordinary.propagate()
    assert ordinary.domains["row_0_0_0"] == tuple(range(9))

    state = network.create_mac_state(domains)
    policy = ClusterClosure(network, ("locked_candidates",), RuleLimits())
    state.on_closure = policy
    keys = set(state._cores)
    assert state.propagate()
    assert state.domains["row_0_0_0"] == (0, 1, 2)
    assert set(state._cores) == keys == set(network.coresDict)
    assert not state._added_core_keys
    assert policy.by_family["locked_candidates"]["useful"] > 0
    # The ordinary state is independent of the experiment's domain restrictions.
    assert ordinary.domains["row_0_0_0"] == tuple(range(9))


def test_cluster_projection_and_context_reset_between_sibling_branches():
    network = ConstraintNetwork(get_sudoku_constraint_network(num=3), maintain_indices=True)
    state = network.create_mac_state()
    state.on_closure = ClusterClosure(network, ("locked_candidates",), RuleLimits())
    assert state.propagate()
    root, keys = state.snapshot_domains(), set(state._cores)
    for box_domain, row in (((0, 1, 2), 0), ((3, 4, 5), 1)):
        state.push_level()
        state.restrict_domain("square_0_0_0", box_domain)
        assert state.propagate()
        assert state.domains[f"row_0_0_{row}"] == (0, 1, 2)
        if row == 1:
            assert 3 in state.domains["row_0_0_0"]
        state.pop_level()
        assert state.snapshot_domains() == root
        assert state.propagate() and state.snapshot_domains() == root
        assert set(state._cores) == keys and not state._added_core_keys
