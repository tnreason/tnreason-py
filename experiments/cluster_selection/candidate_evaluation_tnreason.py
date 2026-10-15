from dataclasses import dataclass
from typing import Callable

from tnreason import engine

from experiments.constraint_networks.generalized_arc_consistency import (
    DomainDict,
    enforce_generalized_arc_consistency,
)


@dataclass(frozen=True)
class SummaryClusterCandidate:
    key: str
    core_keys: tuple[str, ...]
    open_colors: tuple[str, ...]


@dataclass(frozen=True)
class SummaryClusterEvaluation:
    candidate: SummaryClusterCandidate
    domains: DomainDict
    contradiction: bool
    removed_domain_values: int
    cost: float
    reward: float
    summary_key: str


def exponential_open_color_cost(candidate: SummaryClusterCandidate) -> float:
    return float(2 ** len(candidate.open_colors))


def evaluate_summary_cluster_candidate(
    core_dict,
    domains: DomainDict,
    candidate: SummaryClusterCandidate,
    cost_weight: float = 0.05,
    contradiction_reward: float = 0.0,
    cost_function: Callable[[SummaryClusterCandidate], float] = exponential_open_color_cost,
) -> SummaryClusterEvaluation:
    summary_key = f"summary_{candidate.key}"
    selected_cores = {core_key: core_dict[core_key] for core_key in candidate.core_keys}

    updated = dict(core_dict)
    updated[summary_key] = engine.contract(selected_cores, openColors=list(candidate.open_colors))

    candidate_domains = {color: tuple(values) for color, values in domains.items()}
    new_domains, contradiction = enforce_generalized_arc_consistency(
        updated,
        domains=candidate_domains,
        initial_queue=(summary_key,),
    )

    removed_domain_values = sum(
        len(domains[color]) - len(new_domains[color])
        for color in domains
    )
    cost = float(cost_function(candidate))
    reward = float(removed_domain_values) - cost_weight * cost
    if contradiction:
        reward += float(contradiction_reward)

    return SummaryClusterEvaluation(
        candidate=candidate,
        domains=new_domains,
        contradiction=contradiction,
        removed_domain_values=removed_domain_values,
        cost=cost,
        reward=reward,
        summary_key=summary_key,
    )
