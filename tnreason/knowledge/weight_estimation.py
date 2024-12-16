from tnreason import engine
from tnreason import encoding
from tnreason import algorithms

from tnreason.knowledge import distributions as dist
from tnreason.knowledge import deductive as ded

import numpy as np


class WeightEstimator:
    """
    Uses algorithms.MomentMatcher to estimate weights
    """
    def __init__(self, hybridKB, satisfactionDict=None):
        self.hybridKB = hybridKB
        self.satisfactionDict = satisfactionDict

    def get_satisfaction_dict(self, empDistribution):
        empDistributionInferer = ded.InferenceProvider(empDistribution)
        self.satisfactionDict = {key: empDistributionInferer.ask(self.hybridKB.weightedFormulas[key][:-1]) for key
                                 in self.hybridKB.weightedFormulas}

    def fact_check(self):
        """
        Takes the formulas which are always true or false as facts
        """
        for key in list(self.satisfactionDict.keys()):
            assert self.satisfactionDict[key] <= 1 and self.satisfactionDict[key] >= 0, "Empirical satisfaction rate {} of key {} is wrong.".format(
                self.satisfactionDict[key], key)
            if self.satisfactionDict[key] == 0:
                print("Formula {} is never satisfied.".format(self.hybridKB.weightedFormulas[key]))
                formula = self.hybridKB.weightedFormulas.pop(key)
                self.hybridKB.facts[key] = ["not", formula[:-1]]
                self.satisfactionDict.pop(key)
            elif self.satisfactionDict[key] == 1:
                print("Formula {} is always satisfied.".format(self.hybridKB.weightedFormulas[key]))
                formula = self.hybridKB.weightedFormulas.pop(key)
                self.hybridKB.facts[key] = formula[:-1]
                self.satisfactionDict.pop(key)

    def calibrate_weights(self, sweepNum, engineSpec=dict()):
        """
        Optimizes the weights of the remaining formulas
        """
        momentCoreSuffix = encoding.suf.headCoreSuffix
        kbCores = self.hybridKB.create_cores()

        headCores = {}
        for key in self.satisfactionDict:
            headCores[key + momentCoreSuffix] = kbCores.pop(key + momentCoreSuffix)

        calibrator = algorithms.MomentMatcher(structureCores=kbCores,
                                              targetCores={key + momentCoreSuffix: encoding.create_boolean_head(
                                                  color=encoding.get_formula_color(
                                                      self.hybridKB.weightedFormulas[key][:-1]),
                                                  headType="uncertaintyAsWeight",
                                                  weight=self.satisfactionDict[key],
                                                  name=key + momentCoreSuffix)[key+momentCoreSuffix] for key in self.satisfactionDict},
                                              headCores=headCores,
                                              binaryMoments=True,
                                              **engineSpec)
        calibrator.alternating_matching(sweepNum=sweepNum)
        for key in self.satisfactionDict:
            self.hybridKB.weightedFormulas[key][-1] = calibrator.weightDict[key + momentCoreSuffix][-1]

        return calibrator.weightDict

class EntropyMaximizer(engine.EngineUser):
    """
    Optimizes the weights of weighted formulas based on an entropy maximization principle.
    Independent implementation of the special case of two-dimensional exponentiated weights of algorithms.MomentMatcher

    Special case of algorithms.MomentMatcher for local probabilistic interpretation
    -> But: MomentMatcher does directly build the head cores and does not keep track of the weight!
    """

    def __init__(self, expressionsDict, satisfactionDict, backCores={},
                 **engineSpec):
        super().__init__(**engineSpec)
        self.engineSpec = engineSpec

        self.backCores = backCores
        self.expressionsDict = expressionsDict
        self.satisfactionDict = satisfactionDict

        self.formulaCores = encoding.create_formulas_cores({
            key: expressionsDict[key] + [0] for key in expressionsDict
        }, coreType=self.coreType)

        self.check_for_facts()

    def check_for_facts(self):
        factsDict = {}
        for optKey in self.satisfactionDict.keys():
            assert self.satisfactionDict[optKey] <= 1 and self.satisfactionDict[
                optKey] >= 0, "Empirical satisfaction rate {} of key {} is wrong.".format(self.satisfactionDict[optKey],
                                                                                          optKey)
            if self.satisfactionDict[optKey] == 0:
                self.formulaCores.update(
                    encoding.create_formula_head(self.expressionsDict[optKey], headType="falseEvaluation"))
                print("Formula {} is never satisfied.".format(self.expressionsDict[optKey]))
                factsDict[optKey] = 0
            elif self.satisfactionDict[optKey] == 1:
                self.formulaCores.update(
                    encoding.create_formula_head(self.expressionsDict[optKey], headType="truthEvaluation"))
                print("Formula {} is always satisfied.".format(self.expressionsDict[optKey]))
                factsDict[optKey] = 1
        self.factsDict = factsDict

    def alternating_optimization(self, sweepNum=10):
        updateKeys = {key for key in self.satisfactionDict if self.satisfactionDict[key] not in [0, 1]}
        weightDict = {key: [] for key in updateKeys}

        ## Weight optimization
        for sweep in range(sweepNum):
            for optKey in updateKeys:
                local_weight = self.local_condition_satisfier(optKey, self.satisfactionDict[optKey])
                weightDict[optKey].append(local_weight)
                self.formulaCores.update(
                    encoding.create_formula_head(self.expressionsDict[optKey], headType="expFactor",
                                                 weight=local_weight))
        return weightDict, self.factsDict

    def local_condition_satisfier(self, optKey, empRate):
        """
        Returns weight satisfying the local moment matching
        Interpretation as head core: [1 exp[w]]
        """
        optColor = encoding.get_formula_color(self.expressionsDict[optKey])
        tboCoreKey = optColor + encoding.suf.headCoreSuffix
        contracted = engine.contract(coreDict={**self.backCores,
                                               **{key: self.formulaCores[key] for key in self.formulaCores if
                                                  key != tboCoreKey}},
                                     openColors=[optColor], **self.engineSpec)
        negValue = contracted[{optColor: 0}]
        posValue = contracted[{optColor: 1}]
        if negValue != 0 and posValue != 0:
            return np.log((negValue / posValue) * (empRate / (1 - empRate)))
        elif negValue == 0 or posValue == 0:
            return 0  ## In this case the formula is redundant

    def get_optimized_knowledge_base(self, sweepNum=10, updateKeys=None):
        weightDict, factsDict = self.alternating_optimization(sweepNum=sweepNum, updateKeys=updateKeys)
        return dist.HybridKnowledgeBase(
            weightedFormulas={key: self.expressionsDict[key] + [weightDict[key][-1]] for key in weightDict},
            facts={key: self.expressionsDict[key] for key in factsDict},
            backCores=self.backCores
        )
