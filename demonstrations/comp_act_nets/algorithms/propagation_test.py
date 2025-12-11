from demonstrations.comp_act_nets.algorithms import propagation as cp
from demonstrations.comp_act_nets import sudoku_example as se

testNum = 2
assert len(se.create_sudoku_rule_tensor_network(testNum)) == 4 * testNum ** 6

evidenceCores = se.encode_trivial_extended_evidence(
   [(0, 0, 0, 0, 1), (1, 1, 0, 1, 0)], num=2
)
assert len(evidenceCores) == testNum ** 6
assert evidenceCores["0_0_0_0_0_eC"].values[0] == 1
assert evidenceCores["0_0_0_0_0_eC"].values[1] == 1
assert evidenceCores["0_0_0_0_1_eC"].values[0] == 0
assert evidenceCores["0_0_0_0_1_eC"].values[1] == 1