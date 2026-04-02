import unittest

from tnreason import reasoning

from demonstrations.sudoku import sudoku_forward_mp as sol
from demonstrations.sudoku import visualization as vis
from demonstrations.sudoku.examples import standard_sudoku as rep

class SudokuTest(unittest.TestCase):
    def test_num2(self):
        num = 2

        evidenceDict = {
            "a_0_1_0_0_1": 1,
            "a_0_0_0_1_0": 1,
            "a_0_1_1_0_3": 1,
            "a_1_0_0_0_2": 1,
            "a_0_0_1_0_2": 1
        }

        ## Representation ##

        caNetwork = rep.get_assignment_as_CANetwork(num=num, startAssignment=evidenceDict)

        ## Reasoning ##

        # propagator = reasoning.get_inferer("ExpectationPropagator")(
        #     caNetwork=caNetwork,
        #     clusterDict=sol.get_simple_clusterDict(num=num),
        #     meanParamDict=dict()
        # )
        propagator = sol.get_forward_mp_inferer(num, evidenceDict)
        propagator.propagate_until_convergence(evidenceDict.keys())

        solution_array = vis.evidence_to_array(sol.meanParams_to_evidence(propagator.meanParamDict), num=num)
        self.assertTrue(solution_array[0, 0] == 4)
        self.assertTrue(solution_array[0, 1] == 1)
        self.assertTrue(solution_array[1, 0] == 2)
        self.assertTrue(solution_array[1, 1] == 3)
        self.assertTrue(solution_array[0, 2] == 3)
        self.assertTrue(solution_array[0, 3] == 2)
        self.assertTrue(solution_array[1, 2] == 4)

if __name__ == "__main__":
    unittest.main()
