import unittest

from tnreason import reasoning
from tnreason import representation

from tnreason.application import formulas_to_cores as ftc

rules = {
    "f1": ["imp", "a1", "a2"],
}

aSuf = representation.suf.disVarSuf
domainCoreSuffix = representation.suf.eviCoreIn + representation.suf.actCoreSuf

class FCTest(unittest.TestCase):
    def test_modus_ponens(self):
        preEvidence = {
            "a1": 1
        }

        propagator = reasoning.ConstraintPropagator(
            {**ftc.create_cores_to_expressionsDict(rules),
             **ftc.create_formula_evidence_cores(preEvidence)},
            verbose=False
        )
        propagator.propagate_cores()
        assignmentDict = propagator.find_assignments()

        self.assertTrue(assignmentDict["a2" + aSuf] == 1)

    def test_refutation(self):
        preEvidence = {
            "a2": 0
        }

        propagator = reasoning.ConstraintPropagator(
            {**ftc.create_cores_to_expressionsDict(rules),
             **ftc.create_formula_evidence_cores(preEvidence)},
            verbose=False
        )
        propagator.propagate_cores()
        assignmentDict = propagator.find_assignments()

        self.assertTrue(assignmentDict["a1" + aSuf] == 0)

        activationCone = propagator.find_variable_cone(["a1" + aSuf])
        self.assertTrue("a1" + aSuf + domainCoreSuffix in activationCone)
        self.assertTrue(len(activationCone) == 1)

    def test_activationCone_pureImp(self):
        propagator = reasoning.ConstraintPropagator(ftc.create_cores_to_expressionsDict(rules), verbose=False)
        propagator.propagate_cores()
        activationCone = propagator.find_variable_cone(["a1" + aSuf, "a2" + aSuf])

        self.assertTrue(len(activationCone) == 4)
        self.assertTrue("(imp_a1_a2)" + representation.suf.comCoreSuf in activationCone)
        self.assertTrue("a1" + aSuf + domainCoreSuffix in activationCone)
        self.assertTrue("a2" + aSuf + domainCoreSuffix in activationCone)

    def test_activationCone_andFact(self):
        propagator = reasoning.ConstraintPropagator(ftc.create_cores_to_expressionsDict({"r1": ["and", "a1", "a2"]}),
                                                    verbose=False)
        propagator.propagate_cores()
        activationCone = propagator.find_variable_cone(["a1" + aSuf, "a2" + aSuf])

        self.assertTrue(len(activationCone) == 2)
        self.assertTrue("a1" + aSuf + domainCoreSuffix in activationCone)
        self.assertTrue("a2" + aSuf + domainCoreSuffix in activationCone)


if __name__ == "__main__":
    unittest.main()
