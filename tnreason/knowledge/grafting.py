from tnreason import algorithms
from tnreason import encoding
from tnreason import engine

from tnreason.knowledge import distributions as dist
from tnreason.knowledge import deductive as ded

headNeuronString = "headNeurons"
architectureString = "architecture"
acceptanceCriterionString = "acceptanceCriterion"
methodSelectionString = "method"  # Entry in specDict, either one of algorithms.energyOptimizationMethods or klMaximumMethodString
annealingArgumentString = "annealingPattern"  # used in meanField and gibbs

## KLDivergence-based
klMaximumMethodString = "exactKLMax"


def check_boosting_dict(specDict):
    if methodSelectionString not in specDict:
        raise ValueError("Method not specified for Boosting a formula!")
    if headNeuronString not in specDict:
        raise ValueError("Head Neuron not specified for Boosting a formula!")
    if methodSelectionString not in specDict:
        raise ValueError("Architecture is not specified for Boosting a formula!")


class Grafter:
    """
    Searches for best formula by the grafting heuristic: Formulation by an energy optimization problem
    Exceptional handling of KL Divergence: Distinguish between positive and negative phase
    when calculating coordinatewise KL divergence
    """

    def __init__(self, knowledgeBase, specDict):
        self.knowledgeBase = knowledgeBase
        self.specDict = specDict

    def find_candidate(self, empiricalDistribution):
        """
        Searches for a candidate formula
        """
        self.proposalDistribution = dist.ProposalDistribution(
            positivePhase=empiricalDistribution,
            negativePhase=self.knowledgeBase,
            statisticCores=encoding.create_architecture(
                encoding.parse_neuronNameDict_to_neuronColorDict(self.specDict[architectureString]),
                self.specDict[headNeuronString])
        )
        solutionDict = ded.InferenceProvider(self.proposalDistribution).search_mode(
            variableList=encoding.find_selection_colors(self.specDict[architectureString]),
            optimizationMethod=self.specDict.get("method", "numpyArgMax")
        )

        self.candidates = encoding.create_solution_expression(self.specDict[architectureString], solutionDict)

    def test_candidates(self):
        """
        Tests whether to accept the candidate.
        """
        if self.specDict["acceptanceCriterion"] == "always":
            return True
        else:
            raise ValueError("Acceptance Criterion {} not understood.".format(self.specDict[acceptanceCriterionString]))
