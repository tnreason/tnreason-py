from collections import defaultdict, deque
from copy import copy
from typing import Dict, Iterable, Tuple

import numpy as np

from tnreason import engine, representation


DomainDict = Dict[str, Tuple[int, ...]]


############### GAC ###############################

def enforce_generalized_arc_consistency(
    core_dict,
    domains: DomainDict | None = None,
    initial_queue: Iterable[str] | None = None,
) -> tuple[DomainDict, bool]:
    if domains is None:
        domains = initial_domains(core_dict)

    core_keys_by_color = defaultdict(list)
    for core_key, core in core_dict.items():
        for color in core.colors:
            core_keys_by_color[color].append(core_key)

    queue_keys = tuple(core_dict.keys()) if initial_queue is None else tuple(initial_queue)
    queue = deque(queue_keys)
    queued = set(queue_keys)

    while queue:
        core_key = queue.popleft()
        queued.discard(core_key)
        # if core_key not in core_dict:
        #     continue

        core = core_dict[core_key]
        for color in core.colors:
            revised = _supported_domain(core, color, domains)
            if not revised:
                return domains, True

            old_domain = domains[color]
            if len(revised) < len(old_domain):
                domains[color] = revised
                for neighbor_key in core_keys_by_color[color]:
                    if neighbor_key != core_key and neighbor_key not in queued:
                        queue.append(neighbor_key)
                        queued.add(neighbor_key)

    return domains, False


def initial_domains(core_dict) -> DomainDict:
    return {
        color: tuple(range(dim))
        for color, dim in engine.get_dimDict(core_dict).items()
    }


def _supported_domain(core, color: str, domains: DomainDict) -> Tuple[int, ...]:
    axis = core.colors.index(color)
    supported = []
    for value in domains[color]:
        if _slice_has_support(core, axis, value, domains):
            supported.append(value)
    return tuple(supported)


def _slice_has_support(core, axis: int, value: int, domains: DomainDict) -> bool:
    values = getattr(core, "values", None)
    if values is None:
        return _slice_has_support_by_iteration(core, axis, value, domains)

    slices = []
    for index, color in enumerate(core.colors):
        if index == axis:
            slices.append(value)
        else:
            slices.append(list(domains[color]))

    restricted = values[np.ix_(*[
        [item] if isinstance(item, int) else item
        for item in slices
    ])]
    return bool(np.any(restricted != 0))


def _slice_has_support_by_iteration(core, axis: int, value: int, domains: DomainDict) -> bool:
    color = core.colors[axis]
    domain_sets = {domain_color: set(domain) for domain_color, domain in domains.items()}
    for coordinate_value, assignment in copy(core):
        if coordinate_value == 0:
            continue
        if assignment[color] != value:
            continue
        if all(assignment[domain_color] in domain_sets[domain_color] for domain_color in core.colors):
            return True
    return False


##################### to use the contraction mechanism replace ###################
##################### _supported_domain by _supported_domain_by_contraction #################
##################### in enforce_generalized_arc_consistency #######################

def _supported_domain_by_contraction(core, color: str, domains: DomainDict) -> Tuple[int, ...]:
    dim_dict = engine.get_dimDict({"constraint": core})
    local_cores = {"constraint": core}
    for other_color in core.colors:
        if other_color == color:
            continue
        local_cores[f"support_{other_color}"] = _domain_indicator_core(
            color=other_color,
            domain=domains[other_color],
            dimension=dim_dict[other_color],
        )

    message_core = engine.contract(local_cores, openColors=[color])
    return tuple(
        value
        for value in domains[color]
        if message_core[{color: value}] != 0
    )


def _domain_indicator_core(color: str, domain: Tuple[int, ...], dimension: int):
    return engine.create_from_slice_iterator(
        name=f"support_{color}",
        shape=[dimension],
        colors=[color],
        sliceIterator=[
            (1, {color: int(value)})
            for value in domain
        ],
    )

########################### Add domain cores after GAC
########################### to prepare for forward chaining

def add_domain_cores(
    core_dict,
    domains: DomainDict,
    prefix: str = "domain",
    singleton_only: bool = False,
):
    updated = dict(core_dict)
    dim_dict = engine.get_dimDict(core_dict)
    for color, domain in domains.items():
        if color not in dim_dict:
            continue
        domain = tuple(sorted(set(int(value) for value in domain)))
        if singleton_only and len(domain) != 1:
            continue
        if domain == tuple(range(dim_dict[color])):
            continue
        core_key = f"{prefix}_{color}"
        if core_key in updated:
            continue
        if len(domain) == 1:
            updated[core_key] = representation.create_basis_core(
                name=core_key,
                shape=[dim_dict[color]],
                colors=[color],
                numberTuple=[domain[0]],
            )
        else:
            updated[core_key] = engine.create_from_slice_iterator(
                name=core_key,
                shape=[dim_dict[color]],
                colors=[color],
                sliceIterator=[(1, {color: value}) for value in domain],
            )
    return updated



########################### Add larger clusters #############################


def add_cluster_summary_core(core_dict, open_colors: Iterable[str], name: str | None = None):
    open_colors = tuple(open_colors)
    if not open_colors:
        return dict(core_dict)

    relevant = {
        core_key: core
        for core_key, core in core_dict.items()
        if any(color in core.colors for color in open_colors)
    }
    if not relevant:
        return dict(core_dict)

    summary_name = name or "summary_" + "_".join(open_colors)
    updated = dict(core_dict)
    updated[summary_name] = engine.contract(relevant, openColors=list(open_colors))
    return updated
