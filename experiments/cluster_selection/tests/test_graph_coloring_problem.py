from experiments.cluster_selection import evaluate_summary_cluster_candidate
from experiments.cluster_selection.problems.graph_coloring import (
    fixed_graph_coloring_candidates,
    fixed_graph_coloring_core_dict,
    fixed_graph_coloring_domains,
)
from experiments.constraint_networks.generalized_arc_consistency import (
    enforce_generalized_arc_consistency,
)


def _closed_domains(core_dict, domains):
    closed_domains, contradiction = enforce_generalized_arc_consistency(
        core_dict,
        domains={color: tuple(values) for color, values in domains.items()},
    )
    assert not contradiction
    return closed_domains


def _evaluations(core_dict, domains):
    return {
        key: evaluate_summary_cluster_candidate(
            core_dict,
            domains,
            candidate,
            cost_weight=0.01,
            contradiction_reward=100.0,
        )
        for key, candidate in fixed_graph_coloring_candidates().items()
    }


def test_fixed_graph_coloring_has_no_initial_gac_reductions():
    core_dict = fixed_graph_coloring_core_dict()
    domains = _closed_domains(core_dict, fixed_graph_coloring_domains())

    assert all(domain == (0, 1, 2) for domain in domains.values())


def test_restricted_left_odd_cycle_selects_left_cycle_cluster():
    core_dict = fixed_graph_coloring_core_dict()
    domains = _closed_domains(
        core_dict,
        fixed_graph_coloring_domains(restrictions={
            "o0": (0, 1),
            "o1": (0, 1),
            "o2": (0, 1),
            "o3": (0, 1),
            "o4": (0, 1),
        }),
    )

    evaluations = _evaluations(core_dict, domains)

    assert evaluations["left_odd_cycle"].contradiction
    assert not evaluations["right_odd_cycle"].contradiction
    assert evaluations["left_odd_cycle"].reward > evaluations["right_odd_cycle"].reward
    assert evaluations["left_odd_cycle"].reward > evaluations["left_implication"].reward


def test_restricted_right_odd_cycle_selects_right_cycle_cluster():
    core_dict = fixed_graph_coloring_core_dict()
    domains = _closed_domains(
        core_dict,
        fixed_graph_coloring_domains(restrictions={
            "t0": (0, 1),
            "t1": (0, 1),
            "t2": (0, 1),
            "t3": (0, 1),
            "t4": (0, 1),
        }),
    )

    evaluations = _evaluations(core_dict, domains)

    assert evaluations["right_odd_cycle"].contradiction
    assert not evaluations["left_odd_cycle"].contradiction
    assert evaluations["right_odd_cycle"].reward > evaluations["left_odd_cycle"].reward
    assert evaluations["right_odd_cycle"].reward > evaluations["right_implication"].reward


def test_left_assignment_selects_left_implication_cluster():
    core_dict = fixed_graph_coloring_core_dict()
    domains = _closed_domains(core_dict, fixed_graph_coloring_domains(restrictions={"l": (0,)}))

    assert domains["x"] == (0, 1, 2)

    evaluations = _evaluations(core_dict, domains)

    assert not evaluations["left_implication"].contradiction
    assert evaluations["left_implication"].domains["x"] == (0,)
    assert evaluations["left_implication"].reward > evaluations["right_implication"].reward
    assert evaluations["left_implication"].reward > evaluations["left_odd_cycle"].reward


def test_right_assignment_selects_right_implication_cluster():
    core_dict = fixed_graph_coloring_core_dict()
    domains = _closed_domains(core_dict, fixed_graph_coloring_domains(restrictions={"h": (2,)}))

    assert domains["y"] == (0, 1, 2)

    evaluations = _evaluations(core_dict, domains)

    assert not evaluations["right_implication"].contradiction
    assert evaluations["right_implication"].domains["y"] == (2,)
    assert evaluations["right_implication"].reward > evaluations["left_implication"].reward
    assert evaluations["right_implication"].reward > evaluations["right_odd_cycle"].reward
