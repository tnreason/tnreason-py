from experiments.constraint_networks.tests import definite_clauses
from experiments.constraint_networks.tests import modus_ponens
from experiments.constraint_networks.sudoku_tests import fc_test as fct

import os

os.chdir("sudoku_tests")
success, solNet = fct.fc_test_on_puzzle(0)
assert success