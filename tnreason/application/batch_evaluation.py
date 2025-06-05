from tnreason import representation
from tnreason import reasoning

from tnreason.application import formulas_to_cores as ftc

class KnowledgePropagator:
    """
    Evaluates formulas by constraint propagation.
    """

    def __init__(self, knowledgeBase, evidenceDict={}):
        self.atoms = knowledgeBase.distributedVariables
        self.knowledgeCores = {
            **knowledgeBase.create_cores(),
            **ftc.create_atom_evidence_cores(evidenceDict)}

        self.propagator = reasoning.ConstraintPropagator(binaryCoresDict=self.knowledgeCores)

        self.knownHeads = get_evidence_headKeys(evidenceDict) + [key + representation.suf.actCoreSuf for key in knowledgeBase.facts]
#            representation.get_formula_color(knowledgeBase.facts[key]) + representation.suf.headCoreSuffix for key in
#            knowledgeBase.facts]

    def evaluate(self, variables=None):
        if variables is None:
            variables = self.knownHeads
        self.propagator.initialize_domainCoresDict()
        self.propagator.propagate_cores(coreOrder=variables)
        self.entailedDict = self.propagator.find_assignments()
        return self.entailedDict

    def find_carrying_cores(self, variables=None, variablesShape={}):
        if variables is None:
            variables = self.atoms
        return self.propagator.find_variable_cone(variables, {**variablesShape,
                                                              **{variable: 2 for variable in variables if
                                                                 variable not in variablesShape}})

def get_evidence_headKeys(evidenceDict):
    return [ftc.get_formula_color(key) + representation.suf.eviCoreIn + representation.suf.actCoreSuf for key in
            evidenceDict]