from tnreason import engine, representation


def check_consistency(caNetwork, assignment):
    """
    Checks whether the given assignment is consistent with the given CANetwork, by checking if the cores to any feature have a nonvanishing contraction
    """
    for featureKey in caNetwork.featureDict:
        feature = caNetwork.featureDict[featureKey]

        coreDict = {**{coreKey: caNetwork.computationCoreDict[coreKey] for coreKey in feature.affectedComputationCores},
                    **{featureColor: representation.create_basis_core(name=featureColor, shape=[feature.shape[i]],
                                                                      colors=[featureColor],
                                                                      numberTuple=(assignment[featureColor])) for
                       i, featureColor in enumerate(feature.featureColors) if featureColor in assignment}
                    }

        if len(coreDict) > 0:
            if engine.contract(coreDict, openColors=[])[:] == 0:
                return False

    return True


if __name__ == "__main__":
    from demonstrations.sudoku import sudoku_constraints as sc

    caNet = sc.get_assignment_as_CANetwork(num=3, startAssignment={})

    assert check_consistency(caNet, {"a_0_0_0_0_0": 1})  # True
    assert not check_consistency(caNet, {"a_0_0_0_0_0": 1, "pos_0_0_0_0": 4})  # False


    import pandas as pd
    from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence
    path = '/home/schuette/Desktop/AlphaSudoku/tnreason-py-version2(2)/tnreason-py-version2/demonstrations/sudoku/sudoku_bench/sample_data/'
    df = pd.read_parquet(
        f'{path}nikoli100.parquet')
    puzzleNum = 1
    sol = df["solution"][puzzleNum]
    print("sol", sol)
    solution = initial_board_into_evidence(sol)
    assert check_consistency(caNet, solution) 

    sol_false = "1"+sol[1:]
    print("not sol", sol_false, sol_false == sol)
    solution = initial_board_into_evidence(sol_false)
    assert not check_consistency(caNet, solution)  # False
