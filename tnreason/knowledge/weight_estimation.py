from tnreason import encoding
from tnreason import algorithms

from tnreason.knowledge import deductive as ded

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