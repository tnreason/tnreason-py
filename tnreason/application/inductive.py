from tnreason.application import grafting as gf
from tnreason.application import distributions as dist
from tnreason.application import deductive as ded

from tnreason.application import formulas_to_cores as ftc
from tnreason.application import neurons_to_cores as ntc
from tnreason.application import script_transform as st

from tnreason import reasoning
from tnreason import representation

headNeuronString = "headNeurons"
architectureString = "architecture"


def calculate_satisfactionDict(empDistribution, expressionDict, inferenceMethod=None):
    empCaNet = empDistribution.create_caNetwork()

    expressionComputationCores = dict()
    for expressionKey in expressionDict:
        expressionComputationCores.update(ftc.create_formula_computation_cores(expressionDict[expressionKey]))

    empCaNet.include_features(
        featureDict={expressionKey: representation.SoftPartitionFeature(
            featureColors=[ftc.get_formula_headColor(expressionDict[expressionKey])],
            affectedComputationCores=ftc.create_formula_computation_cores(expressionDict[expressionKey]).keys())
            for expressionKey in expressionDict},
        computationCores=expressionComputationCores
    )

    fInferer = reasoning.get_inferer(inferenceMethod)(caNetwork=empCaNet)
    fInferer.infer_meanParams(featureKeys=expressionDict.keys())

    return {expressionKey: fInferer.meanParamDict[expressionKey][1] / (
            fInferer.meanParamDict[expressionKey][0] + fInferer.meanParamDict[expressionKey][1]) for expressionKey
            in expressionDict}


class HybridLearner:
    """
    Intended to use for extending a Knowledge Base based on data.
    Iterating between:
        - structure learning: Inference on proposal distribution to learn new formulas
        - weight estimation: Using the EntropyMaximizer to adjust the weights to the formulas
    """

    def __init__(self, startKB, engineSpec={"coreType": "PandasCore", "contractionMethod": "CorewiseContractor"}):
        """
        startKB a application.HybridKnowledgeBase instance representing the current application to be extended.
        """
        self.knowledgeBase = startKB # To be flexibilized to arbitrary caNetworks
        self.engineSpec = engineSpec

    def get_knowledge_base(self):
        return self.knowledgeBase

    def initialize_proposalDistribution(self, architecture, headNeuron, empiricalDistribution, alternationMethod=None):
        """
        Empirical Distribution: Positive Phase of Learning
        AlternationMethod: When empirical distribution a dataset, can alternate the data to get a negative phase
        """
        if alternationMethod in reasoning.energySamplingMethods:
            sampler = reasoning.get_energy_based_sampler(self.knowledgeBase.get_energy_dict(),
                                                         samplingMethod=alternationMethod,
                                                         startSlices=empiricalDistribution.as_core(),
                                                         colors=empiricalDistribution.distributedVariables,
                                                         **self.engineSpec
                                                         )
        elif alternationMethod in reasoning.coreSamplingMethods:
            sampler = reasoning.get_core_based_sampler(self.knowledgeBase.create_cores(),
                                                       samplingMethod=alternationMethod,
                                                       startSlices=empiricalDistribution.as_core(),
                                                       colors=empiricalDistribution.distributedVariables,
                                                       **self.engineSpec
                                                       )
        elif alternationMethod is None:
            sampler = None
        else:
            raise ValueError("Alternation Method {} not understood!".format(alternationMethod))

        ## Check whether alternation has happended, otherwise use knowledgeBase as negative phase
        if sampler is not None:
            negativePhase = dist.MarkovNetwork({"negPhase": sampler.to_core()}, coreType=self.engineSpec["coreType"],
                                               contractionMethod=self.engineSpec["contractionMethod"])
        else:
            negativePhase = self.knowledgeBase

        self.proposalDistribution = dist.ProposalDistribution(
            positivePhase=empiricalDistribution,
            negativePhase=negativePhase,
            statisticCores=ntc.create_architecture(architecture,
                                                              [headNeuron], coreType=self.engineSpec["coreType"]),
            **self.engineSpec
        )

    def propose_candidate(self, architecture=dict(), **specDict):
        """
        Can only use energy-based methods, since proposal distribution has no core instantiation
        """
        solutionDict = ded.InferenceProvider(self.proposalDistribution).search_mode(
            variableList=ntc.find_selection_colors(architecture),
            **specDict,
            **self.engineSpec
        )
        return st.create_solution_expression(architecture, solutionDict)

    ## OLD -> Should be dropped along application.Grafter! -> Now proposal distribution inference is directly available
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
        print("Learned formulas: {}".format(booster.candidates))
        if booster.test_candidates():
            print("Accepted formulas.")
            self.knowledgeBase.include(
                dist.HybridKnowledgeBase(weightedFormulas={
                    candidateKey + stepName: booster.candidates[candidateKey] + [0] for candidateKey in
                    booster.candidates}))
            if "calibrationSweeps" not in specDict:
                specDict["calibrationSweeps"] = 10
            self.infer_weights_on_data(empDistribution)

    # New based on inference
    def infer_weights_on_data(self, empDistribution, satInferenceMethod="ForwardContractor",
                              calForwardInferenceMethod="ForwardContractor",
                              calInferenceMethod="BackwardAlternator"):
        satisfactionDict = calculate_satisfactionDict(empDistribution,
                                                      {expressionKey: self.knowledgeBase.weightedFormulas[
                                                                          expressionKey][:-1] for expressionKey in
                                                       self.knowledgeBase.weightedFormulas},
                                                      inferenceMethod=satInferenceMethod)

        ## Filter facts
        for featureKey in satisfactionDict:
            if satisfactionDict[featureKey] == 0:
                self.knowledgeBase.facts[featureKey] = ["not", self.knowledgeBase.weightedFormulas[featureKey][:-1]]
                self.knowledgeBase.weightedFormulas.pop(featureKey)
            elif satisfactionDict[featureKey] == 1:
                self.knowledgeBase.facts[featureKey] = self.knowledgeBase.weightedFormulas[featureKey][:-1]
                self.knowledgeBase.weightedFormulas.pop(featureKey)
            else:
                print("Feature {} to be calibrated to match {}.".format(featureKey, satisfactionDict[featureKey]))

        ## Update canonical parameters
        bInferer = reasoning.get_inferer(calInferenceMethod)(caNetwork=self.knowledgeBase.create_caNetwork(),
                                                             forwardInferer=reasoning.get_inferer(
                                                                 calForwardInferenceMethod)(
                                                                 self.knowledgeBase.create_caNetwork()
                                                             ),
                                                             meanParamDict={featureKey: satisfactionDict[featureKey] for
                                                                            featureKey in
                                                                            self.knowledgeBase.weightedFormulas})

        weights = bInferer.alternating_updates(featureKeys=list(self.knowledgeBase.weightedFormulas.keys()))

        for expressionKey in self.knowledgeBase.weightedFormulas:
            self.knowledgeBase.weightedFormulas[expressionKey][-1] = bInferer.caNetwork.canParamDict[expressionKey]

        return weights  # Used only in unittests for investigation
