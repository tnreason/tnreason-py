from experiments.backtracking_search.check_caNetwork import check_consistency
import json

from demonstrations.sudoku import sudoku_constraints as sc

caNet = sc.get_assignment_as_CANetwork(num=3, startAssignment={})

# Check Sudoku representation as a Computation-Activation Network by contraction with
# something that can be True 
assert check_consistency(caNet, {"a_0_0_0_0_0": 1})  # True
# and something that can not happen in a Sudoku
assert not check_consistency(caNet, {"a_0_0_0_0_0": 1, "pos_0_0_0_0": 4})  # False


# Check the result from backtracking search for the puzzle puzzleNum
path = '/home/schuette/Desktop/AlphaSudoku/tnreason-py-version2(2)/tnreason-py-version2/demonstrations/sudoku/sudoku_bench/sample_data/'
puzzleNum = 1
file = f"{path}backtracking_result_{puzzleNum}.json"

with open(file, "r", encoding="utf-8") as f:
    loaded = json.load(f)

assert check_consistency(caNet, loaded) # True