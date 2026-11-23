from tnreason import engine

from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from experiments.constraint_networks.generalized_arc_consistency import DomainDict


def not_equal_core(left: str, right: str, num_colors: int = 3):
    name = _edge_key(left, right)
    return engine.create_from_slice_iterator(
        name=name,
        colors=[left, right],
        shape=[num_colors, num_colors],
        sliceIterator=[
            (1, {left: left_color, right: right_color})
            for left_color in range(num_colors)
            for right_color in range(num_colors)
            if left_color != right_color
        ],
    )


def graph_coloring_core_dict(edges: tuple[tuple[str, str], ...], num_colors: int = 3):
    return {
        _edge_key(left, right): not_equal_core(left, right, num_colors=num_colors)
        for left, right in edges
    }


def fixed_graph_coloring_core_dict(num_colors: int = 3):
    return graph_coloring_core_dict(_fixed_graph_coloring_edges(), num_colors=num_colors)


def fixed_graph_coloring_variables() -> tuple[str, ...]:
    return (
        "o0", "o1", "o2", "o3", "o4",
        "t0", "t1", "t2", "t3", "t4",
        "l", "x", "a", "b", "c", "d",
        "h", "y", "p", "q", "r", "s",
    )


def fixed_graph_coloring_domains(
    num_colors: int = 3,
    restrictions: dict[str, tuple[int, ...]] | None = None,
) -> DomainDict:
    restrictions = restrictions or {}
    full_domain = tuple(range(num_colors))
    return {
        variable: tuple(restrictions.get(variable, full_domain))
        for variable in fixed_graph_coloring_variables()
    }


def fixed_graph_coloring_candidates() -> dict[str, SummaryClusterCandidate]:
    return {
        "left_odd_cycle": SummaryClusterCandidate(
            key="left_odd_cycle",
            core_keys=("o0_o1", "o1_o2", "o2_o3", "o3_o4", "o4_o0"),
            open_colors=("o0", "o1", "o2", "o3", "o4"),
        ),
        "right_odd_cycle": SummaryClusterCandidate(
            key="right_odd_cycle",
            core_keys=("t0_t1", "t1_t2", "t2_t3", "t3_t4", "t4_t0"),
            open_colors=("t0", "t1", "t2", "t3", "t4"),
        ),
        "left_implication": SummaryClusterCandidate(
            key="left_implication",
            core_keys=("x_a", "a_b", "b_c", "c_d", "d_x", "l_a", "l_b", "l_c", "l_d"),
            open_colors=("l", "x", "a", "b", "c", "d"),
        ),
        "right_implication": SummaryClusterCandidate(
            key="right_implication",
            core_keys=("y_p", "p_q", "q_r", "r_s", "s_y", "h_p", "h_q", "h_r", "h_s"),
            open_colors=("h", "y", "p", "q", "r", "s"),
        ),
    }


def _fixed_graph_coloring_edges() -> tuple[tuple[str, str], ...]:
    return (
        ("o0", "o1"),
        ("o1", "o2"),
        ("o2", "o3"),
        ("o3", "o4"),
        ("o4", "o0"),
        ("t0", "t1"),
        ("t1", "t2"),
        ("t2", "t3"),
        ("t3", "t4"),
        ("t4", "t0"),
        ("x", "a"),
        ("a", "b"),
        ("b", "c"),
        ("c", "d"),
        ("d", "x"),
        ("l", "a"),
        ("l", "b"),
        ("l", "c"),
        ("l", "d"),
        ("y", "p"),
        ("p", "q"),
        ("q", "r"),
        ("r", "s"),
        ("s", "y"),
        ("h", "p"),
        ("h", "q"),
        ("h", "r"),
        ("h", "s"),
    )


def _edge_key(left: str, right: str) -> str:
    return f"{left}_{right}"
