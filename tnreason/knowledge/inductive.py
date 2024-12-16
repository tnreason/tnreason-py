from tnreason.knowledge import weight_estimation as wees
from tnreason.knowledge import grafting as gf
from tnreason.knowledge import distributions as dist


class HybridLearner:
    """
    Intended to use for extending a Knowledge Base based on data.
    Iterating between:
        - structure learning: Using the FormulaBooster to learn new formulas
        - weight estimation: Using the EntropyMaximizer to adjust the weights to the formulas
    """

    def __init__(self, startKB):
        """
        startKB a knowledge.HybridKnowledgeBase instance representing the current knowledge to be extended.
        """
        self.hybridKB = startKB

    def get_knowledge_base(self):
        return self.hybridKB

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
        booster = gf.Grafter(self.hybridKB, specDict)
        booster.find_candidate(empDistribution)
        print("Learned formulas: {}".format(booster.candidates))
        if booster.test_candidates():
            print("Accepted formulas.")
            self.hybridKB.include(
                dist.HybridKnowledgeBase(weightedFormulas={
                    candidateKey + stepName: booster.candidates[candidateKey] + [0] for candidateKey in
                    booster.candidates}))
            if "calibrationSweeps" not in specDict:
                specDict["calibrationSweeps"] = 10
            self.calibrate_weights_on_data(specDict, empDistribution)

    def calibrate_weights_on_data(self, specDict, empDistribution, engineSpec=dict()):
        calibrator = wees.WeightEstimator(self.hybridKB)
        calibrator.get_satisfaction_dict(empDistribution)
        calibrator.fact_check()
        calibrator.calibrate_weights(specDict["calibrationSweeps"], engineSpec)
        self.hybridKB = calibrator.hybridKB