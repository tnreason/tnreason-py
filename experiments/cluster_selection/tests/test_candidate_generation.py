import numpy as np
import pytest
from itertools import combinations
from types import SimpleNamespace
from tnreason import engine
from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate, evaluate_summary_cluster_candidate
from experiments.cluster_selection.candidate_generation import ClusterCandidateGenerator
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.generalized_arc_consistency import enforce_generalized_arc_consistency
from experiments.constraint_networks.sudoku_tests.standard_constraints import get_sudoku_constraint_network


def _core(*colors, dimensions=None):
    shape = tuple((dimensions or {}).get(color, 3) for color in colors)
    return engine.get_core("NumpyCore")(values=np.ones(shape, dtype=bool), colors=list(colors))


def _triangle(tail=False):
    cores = {"ab": _core("a", "b"), "bc": _core("b", "c"), "ca": _core("c", "a")}
    if tail:
        cores["cd"] = _core("c", "d")
    return cores


def _signatures(candidates):
    return [(candidate.core_keys, candidate.open_colors) for candidate in candidates]


def _has_cluster(candidates, core_keys, open_colors=None):
    return any(set(candidate.core_keys) == set(core_keys)
               and (open_colors is None or set(candidate.open_colors) == set(open_colors))
               for candidate in candidates)


def _assert_connected(cores, candidate):
    remaining = set(candidate.core_keys)
    first = remaining.pop()
    colors = set(cores[first].colors)
    while remaining:
        adjacent = {key for key in remaining if colors.intersection(cores[key].colors)}
        assert adjacent, "A proposed cluster must be connected."
        remaining.difference_update(adjacent)
        colors.update(color for key in adjacent for color in cores[key].colors)


@pytest.mark.parametrize("as_network", [False, True])
def test_candidates_use_existing_cores_and_leave_valid_colors_open(as_network):
    cores = _triangle(tail=True)
    source = ConstraintNetwork(cores) if as_network else cores
    candidates = ClusterCandidateGenerator(source, max_candidates=100).generate()
    assert isinstance(candidates, tuple) and candidates
    assert all(isinstance(candidate, SummaryClusterCandidate) for candidate in candidates)
    assert len({candidate.key for candidate in candidates}) == len(candidates)
    assert len(set(_signatures(candidates))) == len(candidates)
    for candidate in candidates:
        assert candidate.core_keys and set(candidate.core_keys).issubset(cores)
        assert candidate.core_keys == tuple(sorted(candidate.core_keys))
        assert candidate.open_colors
        assert len(candidate.open_colors) == len(set(candidate.open_colors))
        assert set(candidate.open_colors).issubset(
            {color for key in candidate.core_keys for color in cores[key].colors})
        _assert_connected(cores, candidate)


def test_generator_recovers_cycle_and_connected_expansion():
    cores = _triangle(tail=True)
    candidates = ClusterCandidateGenerator(cores, max_cores=4, max_cycle_length=4,
                                          max_candidates=100).generate(open_colors=("a", "d"))
    assert _has_cluster(candidates, ("ab", "bc", "ca"))
    assert _has_cluster(candidates, cores)
    assert all(set(candidate.open_colors).issubset({"a", "d"}) for candidate in candidates)


def test_generated_candidate_can_strengthen_gac_through_existing_evaluator():
    core_type = engine.get_core("NumpyCore")
    cores = {"ab": core_type(values=np.array([[0, 1], [1, 1]], dtype=bool), colors=["a", "b"]),
             "ac": core_type(values=np.array([[0, 1], [1, 1]], dtype=bool), colors=["a", "c"]),
             "bc": core_type(values=np.array([[0, 1], [1, 0]], dtype=bool), colors=["b", "c"])}
    domains, contradiction = enforce_generalized_arc_consistency(cores)
    assert not contradiction and domains["a"] == (0, 1)
    candidates = ClusterCandidateGenerator(cores).generate(domains, open_colors=("a",))
    candidate = next(candidate for candidate in candidates if set(candidate.core_keys) == set(cores))
    evaluation = evaluate_summary_cluster_candidate(cores, domains, candidate)
    assert evaluation.domains["a"] == (1,)
    assert evaluation.removed_domain_values == 1 and not evaluation.contradiction
    assert set(cores) == {"ab", "ac", "bc"} and domains["a"] == (0, 1)


def test_connected_candidates_do_not_require_a_cycle():
    cores = {"ab": _core("a", "b"), "bc": _core("b", "c"), "cd": _core("c", "d")}
    candidates = ClusterCandidateGenerator(cores, max_cores=3, max_candidates=100).generate(
        open_colors=("a", "d"))
    assert _has_cluster(candidates, cores)


def test_fixed_color_does_not_connect_otherwise_separate_constraints():
    cores = {"ax": _core("a", "x"), "xb": _core("x", "b")}
    generator = ClusterCandidateGenerator(cores, max_candidates=100)
    assert _has_cluster(generator.generate(), cores)
    assert not _has_cluster(generator.generate({"x": (0,)}), cores)


def test_parallel_binary_cores_can_form_a_candidate():
    cores = {"relation_one": _core("a", "b"), "relation_two": _core("a", "b")}
    candidates = ClusterCandidateGenerator(cores, max_candidates=100).generate()
    assert _has_cluster(candidates, cores)


def test_hypercores_use_incidence_connections_without_being_split():
    cores = {"abc": _core("a", "b", "c"), "bcd": _core("b", "c", "d"),
             "de": _core("d", "e")}
    candidates = ClusterCandidateGenerator(cores, max_cores=3, max_candidates=100).generate()
    assert _has_cluster(candidates, ("abc", "bcd"))
    assert _has_cluster(candidates, cores)
    for candidate in candidates:
        _assert_connected(cores, candidate)


def test_candidates_are_deterministic_and_independent_of_dict_insertion_order():
    cores = _triangle(tail=True)
    domains = {color: (0, 1) for color in "abcd"}
    generator = ClusterCandidateGenerator(cores, max_candidates=100)
    first = generator.generate(domains)
    reversed_cores = dict(reversed(tuple(cores.items())))
    reversed_domains = dict(reversed(tuple(domains.items())))
    assert first == generator.generate(domains)
    assert first == ClusterCandidateGenerator(reversed_cores, max_candidates=100).generate(reversed_domains)


def test_generation_does_not_mutate_network_cores_or_domains():
    cores = _triangle(tail=True)
    network = ConstraintNetwork(cores)
    domains = {color: (0, 1) for color in "abcd"}
    values = {key: core.values.copy() for key, core in cores.items()}
    original_domains, original_attributes = dict(domains), dict(vars(network))
    generator = ClusterCandidateGenerator(network)
    generator.generate(domains)
    assert domains == original_domains
    assert vars(network) == original_attributes
    assert network.coresDict is cores
    for key, core in cores.items():
        np.testing.assert_array_equal(core.values, values[key])


def test_changed_colors_focus_on_the_affected_component():
    cores = {**_triangle(), "uv": _core("u", "v"), "vw": _core("v", "w"), "wu": _core("w", "u")}
    generator = ClusterCandidateGenerator(cores, max_candidates=100)
    affected = generator.generate(changed_colors={"u"})
    assert affected
    assert all(set(candidate.core_keys).issubset({"uv", "vw", "wu"}) for candidate in affected)
    assert generator.generate(changed_colors=set()) == ()


def test_current_domains_prioritize_small_unresolved_variables():
    cores = {**_triangle(), "uv": _core("u", "v"), "vw": _core("v", "w"), "wu": _core("w", "u")}
    domains = {color: tuple(range(3)) for color in "abcuvw"}
    domains.update({color: (0, 1) for color in "uvw"})
    candidates = ClusterCandidateGenerator(cores, max_candidates=1).generate(domains)
    assert candidates
    assert set(candidates[0].open_colors).issubset({"u", "v", "w"})


def test_small_generation_and_contraction_budgets_are_enforced():
    cores = _triangle(tail=True)
    generator = ClusterCandidateGenerator(cores, max_cores=2, max_open_colors=1, max_candidates=2)
    candidates = generator.generate()
    assert len(candidates) <= 2
    assert all(len(candidate.core_keys) <= 2 and len(candidate.open_colors) <= 1 for candidate in candidates)
    assert len(generator.generate(max_candidates=1)) <= 1
    assert ClusterCandidateGenerator(cores, max_work=1, max_size=1).generate() == ()


def test_generation_only_inspects_structure_and_never_contracts(monkeypatch):
    def unexpected_contract(*args, **kwargs):
        raise AssertionError("Candidate generation must not execute tensor contractions.")

    monkeypatch.setattr(engine, "contract", unexpected_contract)
    assert ClusterCandidateGenerator(_triangle()).generate()


def test_cost_cache_stays_bounded_across_domain_and_output_changes():
    generator = ClusterCandidateGenerator(_triangle(tail=True), max_steps=8, max_candidates=100)
    signatures = set()
    for size in (1, 2):
        for output in combinations("abcd", size):
            for domains in (None, {"a": (0, 1)}):
                candidates = generator.generate(domains, open_colors=output)
                signatures.update(_signatures(candidates))
                assert len(generator._costs) <= generator.max_steps
    assert len(signatures) > generator.max_steps  # Exercise eviction, not merely a small cache.


@pytest.mark.parametrize("option", ["max_cores", "max_open_colors", "max_cycle_length",
                                   "max_candidates", "max_work", "max_size", "max_steps"])
def test_invalid_constructor_bounds_are_rejected(option):
    with pytest.raises(ValueError):
        ClusterCandidateGenerator(_triangle(), **{option: -1})


@pytest.mark.parametrize("argument", ["domains", "changed_colors", "open_colors"])
def test_unknown_colors_are_rejected(argument):
    value = {"missing": (0, 1)} if argument == "domains" else {"missing"}
    with pytest.raises(KeyError):
        ClusterCandidateGenerator(_triangle()).generate(**{argument: value})


@pytest.mark.parametrize("domain", [(-1,), (3,), (0.5,)])
def test_invalid_domain_values_are_rejected(domain):
    with pytest.raises(ValueError):
        ClusterCandidateGenerator(_triangle()).generate({"a": domain})


def test_contradictory_domains_and_zero_requested_candidates_return_empty_lists():
    generator = ClusterCandidateGenerator(_triangle())
    assert generator.generate({"a": ()}) == ()
    assert generator.generate(max_candidates=0) == ()
    assert ClusterCandidateGenerator({}).generate() == ()


@pytest.mark.parametrize("cores", [
    {"ab": _core("a", "b"), "bc": _core("b", "c", dimensions={"b": 4})},
    {"duplicate": SimpleNamespace(colors=("a", "a"), shape=(3, 3))},
    {"empty_axis": _core("a", dimensions={"a": 0})},
])
def test_invalid_tensor_shapes_are_rejected(cores):
    with pytest.raises(ValueError):
        ClusterCandidateGenerator(cores)


def _sudoku_motif(num, categories, atoms):
    cores = get_sudoku_constraint_network(num=num)
    colors = set(categories) | set(atoms)
    return {key: core for key, core in cores.items() if set(core.colors).issubset(colors)}


def test_generator_recovers_actual_sudoku_pair_links_without_named_rules():
    positions = ("pos_0_0_0_0", "pos_0_0_0_1")
    units = ("row_0_0_0", "row_1_0_0")
    atoms = ("a_0_0_0_0_0", "a_0_0_0_0_1", "a_0_0_0_1_0", "a_0_0_0_1_1")
    cores = _sudoku_motif(2, positions + units, atoms)
    assert len(cores) == 8
    candidates = ClusterCandidateGenerator(cores, max_cores=8, max_cycle_length=8,
                                          max_candidates=100).generate(open_colors=positions + units)
    assert _has_cluster(candidates, cores, positions + units)


@pytest.mark.parametrize("hidden", [False, True], ids=["naked", "hidden"])
def test_whole_nine_by_nine_domains_recover_a_useful_pair_after_gac_closure(hidden):
    positions = ("pos_0_0_0_0", "pos_0_0_1_0")
    units = ("row_0_0_0", "row_1_0_0")
    atoms = ("a_0_0_0_0_0", "a_0_0_0_0_1", "a_0_0_1_0_0", "a_0_0_1_0_1")
    network = ConstraintNetwork(get_sudoku_constraint_network(num=3))
    domains = {position: (0, 1, 2) if hidden else (0, 1) for position in positions}
    if hidden:
        domains.update({f"pos_0_0_{col // 3}_{col % 3}": tuple(range(2, 9))
                        for col in range(9) if col not in (0, 3)})
    state = network.create_mac_state(domains)
    assert state.propagate()
    closed = state.snapshot_domains()
    assert all(closed[position] == ((0, 1, 2) if hidden else (0, 1)) for position in positions)
    selected = {key for key, core in network.coresDict.items()
                if set(core.colors).issubset(set(positions + units + atoms))}
    assert len(selected) == 8
    candidates = ClusterCandidateGenerator(network).generate(closed)
    candidate = next(candidate for candidate in candidates
                     if set(candidate.core_keys) == selected and set(candidate.open_colors) == set(positions + units))
    evaluation = evaluate_summary_cluster_candidate(network.coresDict, closed, candidate)
    assert not evaluation.contradiction and evaluation.removed_domain_values > 0
    assert all(evaluation.domains[position] == (0, 1) for position in positions)
    assert all(evaluation.domains[unit] == (0, 3) for unit in units)
    assert state.snapshot_domains() == closed


@pytest.mark.parametrize("num", [2, 3])
def test_generator_recovers_actual_sudoku_overlap_and_joined_cycle_links(num):
    categories = ("row_0_0_0", "square_0_0_0")
    atoms = tuple(f"a_0_0_0_{col}_0" for col in range(num))
    cores = _sudoku_motif(num, categories, atoms)
    assert len(cores) == 2 * num
    candidates = ClusterCandidateGenerator(cores, max_cores=2 * num, max_cycle_length=8,
                                          max_candidates=100).generate(open_colors=categories)
    assert _has_cluster(candidates, cores, categories)


def test_mixed_boolean_and_nine_value_dimensions_use_full_tensor_cost():
    categories = ("row_0_0_0", "square_0_0_0")
    atoms = ("a_0_0_0_0_0", "a_0_0_0_1_0")
    cores = _sudoku_motif(3, categories, atoms)
    within_budget = ClusterCandidateGenerator(cores, max_size=81, max_candidates=100)
    too_large = ClusterCandidateGenerator(cores, max_size=80, max_candidates=100)
    assert _has_cluster(within_budget.generate(open_colors=categories), cores, categories)
    assert not _has_cluster(too_large.generate(open_colors=categories), cores, categories)
    domains = {color: (0, 1) for color in categories + atoms}
    assert not _has_cluster(too_large.generate(domains, open_colors=categories), cores, categories)


def test_whole_four_by_four_network_produces_a_bounded_list():
    cores = get_sudoku_constraint_network(num=2)
    candidates = ClusterCandidateGenerator(cores).generate()
    assert 0 < len(candidates) <= 32
    assert all(len(candidate.core_keys) <= 10 and len(candidate.open_colors) <= 4 for candidate in candidates)
    assert any(all(not color.startswith("a_") for color in candidate.open_colors) for candidate in candidates)


def test_whole_nine_by_nine_default_list_contains_short_and_pair_sized_cycles():
    cores = get_sudoku_constraint_network(num=3)
    candidates = ClusterCandidateGenerator(cores).generate()
    assert 0 < len(candidates) <= 32
    for length in (4, 8):
        cycles = [candidate for candidate in candidates if len(candidate.core_keys) == length]
        assert any(all(sum(color in cores[key].colors for key in candidate.core_keys) == 2
                       for color in {color for key in candidate.core_keys for color in cores[key].colors})
                   for candidate in cycles)
