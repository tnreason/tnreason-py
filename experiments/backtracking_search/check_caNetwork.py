from tnreason import engine, representation


def _infer_cellwise_atom_complements(caNetwork, assignment):
    """
    For each assigned atom a_r1_r2_c1_c2_n = 1, add all other atoms for the same
    cell as 0 if they are not explicitly assigned.
    """
    completed = dict(assignment)

    atom_keys = [k for k in caNetwork.featureDict if k.startswith("a_")]
    if not atom_keys:
        return completed

    max_n = max(int(key.split("_")[5]) for key in atom_keys)
    digit_count = max_n + 1

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


def check_consistency(caNetwork, assignment):
    """
    Checks whether the given assignment is consistent with the CANetwork by
    validating non-vanishing contractions over each variable neighborhood.
    """
    assignment = _infer_cellwise_atom_complements(caNetwork, assignment)

    # for featureKey in caNetwork.featureDict:
    #     feature = caNetwork.featureDict[featureKey]

    #     compCores = {coreKey: caNetwork.computationCoreDict[coreKey] for coreKey in feature.affectedComputationCores}
    #     featCores = {featureColor: representation.create_basis_core(name=featureColor, shape=[feature.shape[i]],
    #                                                                   colors=[featureColor],
    #                                                                   numberTuple=(assignment[featureColor])) for
    #                    i, featureColor in enumerate(feature.featureColors) if featureColor in assignment}
    #     coreDict = {**compCores,**featCores}

    #     if len(coreDict) > 0:
    #         if engine.contract(coreDict, openColors=[])[:] == 0:
    #             return False

    # return True


    constraint_to_core_keys = {}
    for core_key, core in caNetwork.computationCoreDict.items():
        # In Sudoku constraints, atom colors are a_* and each core also contains
        # one non-atom categorical color (row_*, col_*, square_*, pos_*).
        for color in core.colors:
            if not color.startswith("a_"):
                constraint_to_core_keys.setdefault(color, set()).add(core_key)

    for core_keys in constraint_to_core_keys.values():
        neighborhood_cores = {core_key: caNetwork.computationCoreDict[core_key] for core_key in core_keys}
        relevant_colors = set()
        for core in neighborhood_cores.values():
            relevant_colors.update(core.colors)

        evidence_cores = {}
        for color in relevant_colors:
            if color in assignment:
                evidence_cores[color] = representation.create_basis_core(
                    name=color,
                    shape=caNetwork.featureDict[color].shape,
                    colors=[color],
                    numberTuple=(assignment[color])
                )

        if engine.contract({**neighborhood_cores, **evidence_cores}, openColors=[])[:] == 0:
            return False

    return True


if __name__ == "__main__":
    from demonstrations.sudoku import sudoku_constraints as sc

    caNet = sc.get_assignment_as_CANetwork(num=3, startAssignment={})

    # Check Sudoku representation as a Computation-Activation Network
    assert check_consistency(caNet, {"a_0_0_0_0_0": 1})  # True
    assert not check_consistency(caNet, {"a_0_0_0_0_0": 1, "pos_0_0_0_0": 4})  # False


    # Check if the checker works with known Sudoku solutions and non-solutions
    import pandas as pd
    from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence
    puzzleNum = 1
    path = '/home/schuette/Desktop/AlphaSudoku/tnreason-py-version2(2)/tnreason-py-version2/demonstrations/sudoku/sudoku_bench/sample_data/'
    df = pd.read_parquet(f'{path}nikoli100.parquet')
    
    sol = df["solution"][puzzleNum]
    solution = initial_board_into_evidence(sol)
    assert check_consistency(caNet, solution) 

    sol_false = str((int(sol[0])+1)%9)+sol[1:]
    solution_false = initial_board_into_evidence(sol_false)
    assert not check_consistency(caNet, solution_false)  # False


    # Check if a result from the backtracking search for the puzzle puzzleNum is consistent with the CANetwork
    from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence
    import json

    # Load result from backtracking search for the puzzle puzzleNum
    puzzleNum = 1
    path = '/home/schuette/Desktop/AlphaSudoku/tnreason-py-version2(2)/tnreason-py-version2/demonstrations/sudoku/sudoku_bench/sample_data/'
    file = f"{path}backtracking_result_{puzzleNum}.json"
    with open(file, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    # Check if all Sudoku constraints are fulfilled
    assert check_consistency(caNet, loaded) # True
