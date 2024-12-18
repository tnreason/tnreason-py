from tnreason.knowledge import weight_estimation as wees
from tnreason.knowledge import grafting as gf
from tnreason.knowledge import distributions as dist
from tnreason.knowledge import deductive as ded

from tnreason import algorithms
from tnreason import encoding

headNeuronString = "headNeurons"
architectureString = "architecture"


class HybridLearner:
    """
    Intended to use for extending a Knowledge Base based on data.
    Iterating between:
        - structure learning: Using the FormulaBooster to learn new formulas
        - weight estimation: Using the EntropyMaximizer to adjust the weights to the formulas
    """

    def __init__(self, startKB, engineSpec={"coreType": "PandasCore", "contractionMethod": "CorewiseContractor"}):
        """
        startKB a knowledge.HybridKnowledgeBase instance representing the current knowledge to be extended.
        """
        self.knowledgeBase = startKB
        # self.coreType = coreType
        # self.contractionMethod = contractionMethod
        self.engineSpec = engineSpec

    def get_knowledge_base(self):
        return self.knowledgeBase

    def initialize_proposalDistribution(self, architecture, headNeuron, empiricalDistribution, alternationMethod=None):
        """
        Empirical Distribution: Positive Phase of Learning
        AlternationMethod: When empirical distribution a dataset, can
        """
        if alternationMethod in algorithms.energySamplingMethods:
            sampler = algorithms.get_energy_based_sampler(self.knowledgeBase.get_energy_dict(),
                                                          samplingMethod=alternationMethod,
                                                          startSlices=empiricalDistribution.as_core(),
                                                          colors=empiricalDistribution.distributedVariables,
                                                          **self.engineSpec
                                                          )
        elif alternationMethod in algorithms.coreSamplingMethods:
            sampler = algorithms.get_core_based_sampler(self.knowledgeBase.create_cores(),
                                                        samplingMethod=alternationMethod,
                                                        startSlices=empiricalDistribution.as_core(),
                                                        colors=empiricalDistribution.distributedVariables,
                                                        **self.engineSpec
                                                        )
        elif alternationMethod is None:
            sampler = None
        else:
            raise ValueError("Alternation Method {} not understood!".format(alternationMethod))

        if sampler is not None:
            negativePhase = dist.MarkovNetwork({"negPhase": sampler.to_core()}, coreType=self.engineSpec["coreType"],
                                               contractionMethod=self.engineSpec["contractionMethod"])
        else:
            negativePhase = self.knowledgeBase

        self.proposalDistribution = dist.ProposalDistribution(
            positivePhase=empiricalDistribution,
            negativePhase=negativePhase,
            statisticCores=encoding.create_architecture(
                encoding.parse_neuronNameDict_to_neuronColorDict(architecture),
                [headNeuron], coreType=self.engineSpec["coreType"]),
            **self.engineSpec
        )

    def propose_candidate(self, architecture=dict(), **specDict):
        """
        Can only use energy-based methods, since proposal distribution has no core instantiation
        """
        solutionDict = ded.InferenceProvider(self.proposalDistribution).search_mode(
            variableList=encoding.find_selection_colors(architecture),
            **specDict,
            **self.engineSpec
        )
        return encoding.create_solution_expression(architecture, solutionDict)

    ## OLD -> Should be dropped along knowledge.Grafter!
    def graft_formula(self, specDict, empDistribution, stepName="_grafted"):
        """
        Grafting with
        * specDict: Dictionary specification of the Hyperparameters of the Boosting Step:
            - method: Method for structure learning: als or gibbs supported
            - sweeps: Number of sweeps in structure learning
            - architecture: Collection of neurons
            - headNeurons: List of neuronKeys to be used for formula heads
            - calibrationSweeps: Number of sweeps in weight estimation
        * empDistribution: storing the data used for the boosting step
        * stepName: Specifies a name suffix for the learned formula to be stored in the HybridKnowledgeBase.
                    Needs to differ for each Step to avoid key conflicts.
        """
        booster = gf.Grafter(self.knowledgeBase, specDict)
        booster.find_candidate(empDistribution)
        print(booster.candidates)
        print("Learned formulas: {}".format(booster.candidates))
        if booster.test_candidates():
            print("Accepted formulas.")
            self.knowledgeBase.include(
                dist.HybridKnowledgeBase(weightedFormulas={
                    candidateKey + stepName: booster.candidates[candidateKey] + [0] for candidateKey in
                    booster.candidates}))
            if "calibrationSweeps" not in specDict:
                specDict["calibrationSweeps"] = 10
            self.calibrate_weights_on_data(specDict, empDistribution)

    def calibrate_weights_on_data(self, specDict, empDistribution, engineSpec=dict()):
        calibrator = wees.WeightEstimator(self.knowledgeBase)
        calibrator.get_satisfaction_dict(empDistribution)
        calibrator.fact_check()
        calibrator.calibrate_weights(specDict["calibrationSweeps"], engineSpec)
        self.knowledgeBase = calibrator.hybridKB
