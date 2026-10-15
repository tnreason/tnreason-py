from experiments.cluster_selection import evaluate_summary_cluster_candidate
from experiments.cluster_selection.problems import repeated_graph_coloring
from experiments.cluster_selection.problems.repeated_graph_coloring import (
    RepeatedGraphColoringConfig,
    cycle_two_color_restrictions,
    implication_trigger_restrictions,
    sample_repeated_graph_coloring_state,
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


def _evaluations(core_dict, domains, config):
    return {
        key: evaluate_summary_cluster_candidate(
            core_dict,
            domains,
            candidate,
            cost_weight=0.01,
            contradiction_reward=100.0,
        )
        for key, candidate in repeated_graph_coloring.candidates(config).items()
    }


def _best_key(evaluations):
    return max(evaluations, key=lambda key: evaluations[key].reward)


def test_repeated_graph_coloring_builds_larger_fixed_problem():
    config = RepeatedGraphColoringConfig(
        implication_count=4,
        cycle_lengths=(5, 5, 5, 4, 4),
    )
    core_dict = repeated_graph_coloring.core_dict(config)
    candidates = repeated_graph_coloring.candidates(config)
    domains = _closed_domains(core_dict, repeated_graph_coloring.domains(config))

    assert len(repeated_graph_coloring.variables(config)) == 47
    assert len(core_dict) == 67
    assert len(candidates) == 9
    assert _is_connected(repeated_graph_coloring.edges(config), repeated_graph_coloring.variables(config))
    assert all(variable.startswith("v") for variable in repeated_graph_coloring.variables(config))
    assert all(domain == (0, 1, 2) for domain in domains.values())


def test_implication_trigger_changes_best_candidate_on_same_repeated_graph():
    config = RepeatedGraphColoringConfig(
        implication_count=4,
        cycle_lengths=(5, 5, 5, 4, 4),
    )
    core_dict = repeated_graph_coloring.core_dict(config)
    domains = _closed_domains(
        core_dict,
        repeated_graph_coloring.domains(
            config,
            restrictions=implication_trigger_restrictions(index=2, color=1),
        ),
    )

    evaluations = _evaluations(core_dict, domains, config)
    target = repeated_graph_coloring.candidates(config)["implication_2"].open_colors[1]

    assert evaluations["implication_2"].domains[target] == (1,)
    assert _best_key(evaluations) == "implication_2"


def test_cycle_restriction_changes_best_candidate_on_same_repeated_graph():
    config = RepeatedGraphColoringConfig(
        implication_count=4,
        cycle_lengths=(5, 5, 5, 4, 4),
    )
    core_dict = repeated_graph_coloring.core_dict(config)
    domains = _closed_domains(
        core_dict,
        repeated_graph_coloring.domains(
            config,
            restrictions=cycle_two_color_restrictions(index=1, colors=(0, 2), config=config),
        ),
    )

    evaluations = _evaluations(core_dict, domains, config)

    assert evaluations["cycle_1"].contradiction
    assert _best_key(evaluations) == "cycle_1"


def test_sample_repeated_graph_coloring_state_returns_fixed_graph_domains_and_label():
    config = RepeatedGraphColoringConfig(
        implication_count=4,
        cycle_lengths=(5, 5, 5, 4, 4),
    )
    core_dict = repeated_graph_coloring.core_dict(config)
    candidates = repeated_graph_coloring.candidates(config)
    sample = sample_repeated_graph_coloring_state(config, seed=7)
    domains = _closed_domains(core_dict, sample.domains)
    evaluations = _evaluations(core_dict, domains, config)

    assert sample.preferred_candidate_keys
    assert set(sample.preferred_candidate_keys) <= set(candidates)
    assert _best_key(evaluations) in sample.preferred_candidate_keys


def test_repeated_graph_coloring_exports_visualization_dot(tmp_path):
    config = RepeatedGraphColoringConfig(
        implication_count=2,
        cycle_lengths=(5, 4),
    )
    domains = repeated_graph_coloring.domains(
        config,
        restrictions=implication_trigger_restrictions(index=1, color=2),
    )

    dot = repeated_graph_coloring.to_dot(config, domains=domains)
    output_path = repeated_graph_coloring.write_dot(tmp_path / "graph.dot", config, domains=domains)

    assert dot.startswith("graph repeated_graph_coloring")
    assert "cluster_implication_1" in dot
    assert "cluster_cycle_0" in dot
    assert 'style=dashed' in dot
    assert "{2}" in dot
    assert output_path.read_text(encoding="utf-8") == dot


def test_repeated_graph_coloring_exports_networkx_graph():
    config = RepeatedGraphColoringConfig(
        implication_count=2,
        cycle_lengths=(5, 4),
    )
    graph = repeated_graph_coloring.to_networkx_graph(config)

    assert graph.number_of_nodes() == len(repeated_graph_coloring.variables(config))
    assert graph.number_of_edges() == len(repeated_graph_coloring.edges(config))
    assert graph.has_edge("v2", "v8")


def _is_connected(edges, variables):
    neighbors = {variable: set() for variable in variables}
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)

    start = next(iter(variables))
    seen = {start}
    frontier = [start]
    while frontier:
        current = frontier.pop()
        for neighbor in neighbors[current]:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            frontier.append(neighbor)
    return len(seen) == len(variables)
