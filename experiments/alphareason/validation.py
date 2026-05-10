from tnreason import engine, representation


def check_assignment_consistency(ca_network, assignment):
    constraint_to_core_keys = {}
    for core_key, core in ca_network.computationCoreDict.items():
        for color in core.colors:
            constraint_to_core_keys.setdefault(color, set()).add(core_key)

    for core_keys in constraint_to_core_keys.values():
        neighborhood_cores = {core_key: ca_network.computationCoreDict[core_key] for core_key in core_keys}
        relevant_colors = set()
        for core in neighborhood_cores.values():
            relevant_colors.update(core.colors)

        evidence_cores = {}
        for color in relevant_colors:
            if color in assignment:
                evidence_cores[color] = representation.create_basis_core(
                    name=color,
                    shape=ca_network.featureDict[color].shape,
                    colors=[color],
                    numberTuple=(assignment[color]),
                )

        if engine.contract({**neighborhood_cores, **evidence_cores}, openColors=[])[:] == 0:
            return False

    return True


def validate_assignment(ca_network, assignment):
    ok = check_assignment_consistency(ca_network, assignment)
    return ok, int(ok)

