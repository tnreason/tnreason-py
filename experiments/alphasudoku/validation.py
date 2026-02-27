from tnreason import engine, representation


def _infer_cellwise_atom_complements(ca_network, assignment):
    completed = dict(assignment)

    atom_keys = [key for key in ca_network.featureDict if key.startswith("a_")]
    if not atom_keys:
        return completed

    max_digit = max(int(key.split("_")[5]) for key in atom_keys)
    digit_count = max_digit + 1

    for key, value in list(assignment.items()):
        if not (key.startswith("a_") and value == 1):
            continue
        r1, r2, c1, c2, n = key.split("_")[1:]
        for other_n in range(digit_count):
            if other_n == int(n):
                continue
            other_key = f"a_{r1}_{r2}_{c1}_{c2}_{other_n}"
            if other_key not in completed:
                completed[other_key] = 0
    return completed


def check_assignment_consistency(ca_network, assignment):
    assignment = _infer_cellwise_atom_complements(ca_network, assignment)

    constraint_to_core_keys = {}
    for core_key, core in ca_network.computationCoreDict.items():
        for color in core.colors:
            if not color.startswith("a_"):
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
