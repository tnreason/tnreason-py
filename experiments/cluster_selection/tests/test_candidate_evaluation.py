from tnreason import engine

from experiments.cluster_selection import (
    SummaryClusterCandidate,
    evaluate_summary_cluster_candidate,
)
from experiments.constraint_networks.generalized_arc_consistency import (
    enforce_generalized_arc_consistency,
)


def _binary_relation_core(name, left, right, pairs):
    return engine.create_from_slice_iterator(
        name=name,
        colors=[left, right],
        shape=[2, 2],
        sliceIterator=[
            (1, {left: left_value, right: right_value})
            for left_value, right_value in pairs
        ],
    )


def _toy_cluster_selection_core_dict():
    or_relation = ((0, 1), (1, 0), (1, 1))
    neq_relation = ((0, 1), (1, 0))
    eq_relation = ((0, 0), (1, 1))

    return {
        "ab": _binary_relation_core("ab", "a", "b", or_relation),
        "ac": _binary_relation_core("ac", "a", "c", or_relation),
        "bc": _binary_relation_core("bc", "b", "c", neq_relation),
        "ad": _binary_relation_core("ad", "a", "d", eq_relation),
        "de": _binary_relation_core("de", "d", "e", eq_relation),
        "pq": _binary_relation_core("pq", "p", "q", or_relation),
        "pr": _binary_relation_core("pr", "p", "r", or_relation),
        "qr": _binary_relation_core("qr", "q", "r", neq_relation),
        "uv": _binary_relation_core("uv", "u", "v", eq_relation),
        "vw": _binary_relation_core("vw", "v", "w", eq_relation),
    }


def test_summary_cluster_reward_prefers_minimal_useful_cluster():
    core_dict = _toy_cluster_selection_core_dict()
    base_domains, contradiction = enforce_generalized_arc_consistency(core_dict)

    assert not contradiction
    assert all(domain == (0, 1) for domain in base_domains.values())

    candidates = {
        "small_ab": SummaryClusterCandidate("small_ab", ("ab",), ("a", "b")),
        "small_ac": SummaryClusterCandidate("small_ac", ("ac",), ("a", "c")),
        "small_bc": SummaryClusterCandidate("small_bc", ("bc",), ("b", "c")),
        "medium_ab_ac": SummaryClusterCandidate(
            "medium_ab_ac",
            ("ab", "ac"),
            ("a", "b", "c"),
        ),
        "right_abc": SummaryClusterCandidate(
            "right_abc",
            ("ab", "ac", "bc"),
            ("a", "b", "c"),
        ),
        "oversized_abcd": SummaryClusterCandidate(
            "oversized_abcd",
            ("ab", "ac", "bc", "ad"),
            ("a", "b", "c", "d"),
        ),
        "oversized_abcde": SummaryClusterCandidate(
            "oversized_abcde",
            ("ab", "ac", "bc", "ad", "de"),
            ("a", "b", "c", "d", "e"),
        ),
        "wrong_pqr": SummaryClusterCandidate(
            "wrong_pqr",
            ("pq", "pr", "qr"),
            ("p", "q", "r"),
        ),
        "decoy_uvw": SummaryClusterCandidate(
            "decoy_uvw",
            ("uv", "vw"),
            ("u", "v", "w"),
        ),
    }

    evaluations = {
        key: evaluate_summary_cluster_candidate(core_dict, base_domains, candidate)
        for key, candidate in candidates.items()
    }

    assert evaluations["small_ab"].removed_domain_values == 0
    assert evaluations["small_ac"].removed_domain_values == 0
    assert evaluations["small_bc"].removed_domain_values == 0
    assert evaluations["medium_ab_ac"].removed_domain_values == 0
    assert evaluations["decoy_uvw"].removed_domain_values == 0

    assert evaluations["right_abc"].removed_domain_values == 3
    assert evaluations["right_abc"].domains["a"] == (1,)
    assert evaluations["right_abc"].domains["d"] == (1,)
    assert evaluations["right_abc"].domains["e"] == (1,)

    assert evaluations["wrong_pqr"].removed_domain_values == 1
    assert evaluations["wrong_pqr"].domains["p"] == (1,)

    assert evaluations["oversized_abcd"].removed_domain_values == 3
    assert evaluations["oversized_abcde"].removed_domain_values == 3
    assert evaluations["right_abc"].reward > evaluations["oversized_abcd"].reward
    assert evaluations["right_abc"].reward > evaluations["oversized_abcde"].reward
    assert evaluations["right_abc"].reward > evaluations["wrong_pqr"].reward
    assert evaluations["wrong_pqr"].reward > evaluations["medium_ab_ac"].reward

    print("Evaluations:")
    for key, eval in evaluations.items():
        print(f"  {key}: {eval.reward}")