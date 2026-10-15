from dataclasses import dataclass
from pathlib import Path
from random import Random

from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from experiments.constraint_networks.generalized_arc_consistency import DomainDict

from .graph_coloring import graph_coloring_core_dict


@dataclass(frozen=True)
class RepeatedGraphColoringConfig:
    implication_count: int = 8
    cycle_lengths: tuple[int, ...] = (5, 5, 5, 5, 5, 5, 4, 4, 4, 4)
    num_colors: int = 3


@dataclass(frozen=True)
class RepeatedGraphColoringSample:
    domains: DomainDict
    preferred_candidate_keys: tuple[str, ...]
    description: str


def core_dict(config: RepeatedGraphColoringConfig | None = None):
    config = config or RepeatedGraphColoringConfig()
    return graph_coloring_core_dict(
        edges(config),
        num_colors=config.num_colors,
    )


def edges(
    config: RepeatedGraphColoringConfig | None = None,
) -> tuple[tuple[str, str], ...]:
    config = config or RepeatedGraphColoringConfig()
    edges = []
    for index in range(config.implication_count):
        edges.extend(_implication_edges(index))
    for index, cycle_length in enumerate(config.cycle_lengths):
        edges.extend(_cycle_edges(_cycle_variables(index, cycle_length, config)))
    edges.extend(_bridge_edges(config))
    return tuple(edges)


def variables(
    config: RepeatedGraphColoringConfig | None = None,
) -> tuple[str, ...]:
    config = config or RepeatedGraphColoringConfig()
    variables = []
    for index in range(config.implication_count):
        variables.extend(_implication_variables(index))
    for index, cycle_length in enumerate(config.cycle_lengths):
        variables.extend(_cycle_variables(index, cycle_length, config))
    return tuple(variables)


def domains(
    config: RepeatedGraphColoringConfig | None = None,
    restrictions: dict[str, tuple[int, ...]] | None = None,
) -> DomainDict:
    config = config or RepeatedGraphColoringConfig()
    restrictions = restrictions or {}
    full_domain = tuple(range(config.num_colors))
    return {
        variable: tuple(restrictions.get(variable, full_domain))
        for variable in variables(config)
    }


def candidates(
    config: RepeatedGraphColoringConfig | None = None,
) -> dict[str, SummaryClusterCandidate]:
    config = config or RepeatedGraphColoringConfig()
    candidates = {}
    for index in range(config.implication_count):
        key = f"implication_{index}"
        candidates[key] = SummaryClusterCandidate(
            key=key,
            core_keys=tuple(_edge_key(left, right) for left, right in _implication_edges(index)),
            open_colors=_implication_variables(index),
        )
    for index, cycle_length in enumerate(config.cycle_lengths):
        key = f"cycle_{index}"
        variables = _cycle_variables(index, cycle_length, config)
        candidates[key] = SummaryClusterCandidate(
            key=key,
            core_keys=tuple(_edge_key(left, right) for left, right in _cycle_edges(variables)),
            open_colors=variables,
        )
    return candidates


def to_networkx_graph(config: RepeatedGraphColoringConfig | None = None):
    config = config or RepeatedGraphColoringConfig()
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(variables(config))
    graph.add_edges_from(edges(config))
    return graph


def to_dot(
    config: RepeatedGraphColoringConfig | None = None,
    domains: DomainDict | None = None,
    include_candidate_subgraphs: bool = True,
) -> str:
    config = config or RepeatedGraphColoringConfig()
    domains = domains or {}
    candidate_dict = candidates(config)
    candidate_edge_keys = {
        core_key
        for candidate in candidate_dict.values()
        for core_key in candidate.core_keys
    }

    lines = [
        "graph repeated_graph_coloring {",
        '  graph [overlap=false, splines=true];',
        '  node [shape=circle, fixedsize=true, width=0.45, fontsize=10];',
        '  edge [color="#666666"];',
    ]

    for variable in variables(config):
        lines.append(f"  {_dot_quote(variable)}{_dot_node_attributes(variable, domains, config)};")

    if include_candidate_subgraphs:
        for candidate in candidate_dict.values():
            lines.append(f"  subgraph cluster_{_dot_id(candidate.key)} {{")
            lines.append(f"    label={_dot_quote(candidate.key)};")
            lines.append('    color="#dddddd";')
            for variable in candidate.open_colors:
                lines.append(f"    {_dot_quote(variable)};")
            lines.append("  }")

    for left, right in edges(config):
        edge_key = _edge_key(left, right)
        attributes = "" if edge_key in candidate_edge_keys else ' [style=dashed, color="#999999"]'
        lines.append(f"  {_dot_quote(left)} -- {_dot_quote(right)}{attributes};")

    lines.append("}")
    return "\n".join(lines) + "\n"


def write_dot(
    path: str | Path,
    config: RepeatedGraphColoringConfig | None = None,
    domains: DomainDict | None = None,
    include_candidate_subgraphs: bool = True,
) -> Path:
    output_path = Path(path)
    output_path.write_text(
        to_dot(
            config=config,
            domains=domains,
            include_candidate_subgraphs=include_candidate_subgraphs,
        ),
        encoding="utf-8",
    )
    return output_path


def plot(
    path: str | Path | None = None,
    config: RepeatedGraphColoringConfig | None = None,
    domains: DomainDict | None = None,
    seed: int = 0,
):
    config = config or RepeatedGraphColoringConfig()
    domains = domains or {}

    import matplotlib.pyplot as plt
    import networkx as nx

    graph = to_networkx_graph(config)
    position = nx.spring_layout(graph, seed=seed)
    candidate_edge_keys = {
        core_key
        for candidate in candidates(config).values()
        for core_key in candidate.core_keys
    }
    solid_edges = []
    bridge_edges = []
    for left, right in graph.edges:
        edge_key = _edge_key(left, right)
        if edge_key in candidate_edge_keys:
            solid_edges.append((left, right))
        else:
            bridge_edges.append((left, right))

    full_domain = tuple(range(config.num_colors))
    node_colors = [
        "#fff2a8" if tuple(domains.get(variable, full_domain)) != full_domain else "#dbeafe"
        for variable in graph.nodes
    ]
    labels = {
        variable: _node_plot_label(variable, domains, config)
        for variable in graph.nodes
    }

    figure, axis = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(graph, position, node_color=node_colors, node_size=650, ax=axis)
    nx.draw_networkx_edges(graph, position, edgelist=solid_edges, edge_color="#555555", width=1.4, ax=axis)
    nx.draw_networkx_edges(
        graph,
        position,
        edgelist=bridge_edges,
        edge_color="#999999",
        width=1.0,
        style="dashed",
        ax=axis,
    )
    nx.draw_networkx_labels(graph, position, labels=labels, font_size=8, ax=axis)
    axis.set_axis_off()
    figure.tight_layout()

    if path is None:
        plt.show()
        return None

    output_path = Path(path)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def implication_trigger_restrictions(index: int, color: int) -> dict[str, tuple[int, ...]]:
    trigger, *_ = _implication_variables(index)
    return {trigger: (int(color),)}


def cycle_two_color_restrictions(
    index: int,
    colors: tuple[int, int] = (0, 1),
    config: RepeatedGraphColoringConfig | None = None,
) -> dict[str, tuple[int, ...]]:
    config = config or RepeatedGraphColoringConfig()
    color_domain = tuple(sorted(int(color) for color in colors))
    return {
        variable: color_domain
        for variable in _cycle_variables(index, config.cycle_lengths[index], config)
    }


def sample_repeated_graph_coloring_state(
    config: RepeatedGraphColoringConfig | None = None,
    seed: int | None = None,
) -> RepeatedGraphColoringSample:
    config = config or RepeatedGraphColoringConfig()
    rng = Random(seed)
    sample_kind = rng.choice(("implication", "cycle"))

    if sample_kind == "implication":
        index = rng.randrange(config.implication_count)
        color = rng.randrange(config.num_colors)
        restrictions = implication_trigger_restrictions(index, color)
        return RepeatedGraphColoringSample(
            domains=domains(config, restrictions=restrictions),
            preferred_candidate_keys=(f"implication_{index}",),
            description=f"implication_{index}_trigger_{color}",
        )

    contradiction_cycle_indices = [
        index
        for index, cycle_length in enumerate(config.cycle_lengths)
        if cycle_length % 2 == 1
    ]
    if not contradiction_cycle_indices:
        raise ValueError("sample_repeated_graph_coloring_state requires at least one odd-length cycle.")
    index = rng.choice(contradiction_cycle_indices)
    color_pair = tuple(sorted(rng.sample(range(config.num_colors), 2)))
    restrictions = cycle_two_color_restrictions(index, colors=color_pair, config=config)
    return RepeatedGraphColoringSample(
        domains=domains(config, restrictions=restrictions),
        preferred_candidate_keys=(f"cycle_{index}",),
        description=f"cycle_{index}_restricted_{color_pair[0]}_{color_pair[1]}",
    )


def _implication_variables(index: int) -> tuple[str, ...]:
    start = 6 * index
    return (
        _variable_key(start),
        _variable_key(start + 1),
        _variable_key(start + 2),
        _variable_key(start + 3),
        _variable_key(start + 4),
        _variable_key(start + 5),
    )


def _implication_edges(index: int) -> tuple[tuple[str, str], ...]:
    trigger, target, a, b, c, d = _implication_variables(index)
    return (
        (target, a),
        (a, b),
        (b, c),
        (c, d),
        (d, target),
        (trigger, a),
        (trigger, b),
        (trigger, c),
        (trigger, d),
    )


def _cycle_variables(
    index: int,
    cycle_length: int,
    config: RepeatedGraphColoringConfig,
) -> tuple[str, ...]:
    start = _cycle_start(index, config)
    return tuple(_variable_key(start + position) for position in range(cycle_length))


def _cycle_edges(variables: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (variables[position], variables[(position + 1) % len(variables)])
        for position in range(len(variables))
    )


def _bridge_edges(config: RepeatedGraphColoringConfig) -> tuple[tuple[str, str], ...]:
    representatives = _region_representatives(config)
    return tuple(
        (representatives[index], representatives[index + 1])
        for index in range(len(representatives) - 1)
    )


def _region_representatives(config: RepeatedGraphColoringConfig) -> tuple[str, ...]:
    representatives = []
    for index in range(config.implication_count):
        representatives.append(_implication_variables(index)[2])
    for index, cycle_length in enumerate(config.cycle_lengths):
        representatives.append(_cycle_variables(index, cycle_length, config)[0])
    return tuple(representatives)


def _cycle_start(index: int, config: RepeatedGraphColoringConfig) -> int:
    return 6 * config.implication_count + sum(config.cycle_lengths[:index])


def _edge_key(left: str, right: str) -> str:
    return f"{left}_{right}"


def _variable_key(index: int) -> str:
    return f"v{index}"


def _dot_node_attributes(
    variable: str,
    domains: DomainDict,
    config: RepeatedGraphColoringConfig,
) -> str:
    domain = domains.get(variable)
    if domain is None or tuple(domain) == tuple(range(config.num_colors)):
        return ""
    label = f"{variable}\\n{{{','.join(str(value) for value in domain)}}}"
    return f' [label={_dot_quote(label)}, style=filled, fillcolor="#fff2a8"]'


def _dot_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _dot_id(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _node_plot_label(
    variable: str,
    domains: DomainDict,
    config: RepeatedGraphColoringConfig,
) -> str:
    domain = domains.get(variable)
    if domain is None or tuple(domain) == tuple(range(config.num_colors)):
        return variable
    return f"{variable}\n{{{','.join(str(value) for value in domain)}}}"


if __name__ == "__main__":
    config = RepeatedGraphColoringConfig()
    sample = sample_repeated_graph_coloring_state(config, seed=7)
    dot_path = Path.cwd() / "repeated_graph_coloring.dot"
    png_path = Path.cwd() / "repeated_graph_coloring.png"

    write_dot(dot_path, config=config, domains=sample.domains)
    plot(png_path, config=config, domains=sample.domains)

    print(f"sample: {sample.description}")
    print(f"preferred candidates: {', '.join(sample.preferred_candidate_keys)}")
    print(f"wrote {dot_path}")
    print(f"wrote {png_path}")
