from tnreason import engine

from tnreason.representation import basis_calculus as bc
from tnreason.representation import features as ft


class ComputationActivationNetwork(engine.EngineUser):
    def __init__(self, featureDict, computationCoreDict=dict(), baseMeasureCoreDict=dict(), canParamDict=dict(),
                 distributedVariables=[],
                 **engineSpec):
        """
        * featureSpecDict: {featureKey : ExpDistFeature for all features}
        * interpretationDict: {featureKey : List of values} for all features
        """

        super().__init__(**engineSpec)

        self.featureDict = featureDict
        for featureKey in self.featureDict:
            self.featureDict[featureKey].name = featureKey

        self.computationCoreDict = computationCoreDict
        self.baseMeasureCoreDict = baseMeasureCoreDict

        self.canParamDict = canParamDict

        for featureKey in self.featureDict:
            if featureKey not in self.canParamDict: # and type(self.featureDict[featureKey]) != ft.PassiveFeature:
                self.canParamDict[featureKey] = self.featureDict[featureKey].find_neutral_canParam()
            for coreKey in self.featureDict[featureKey].affectedComputationCores:
                if coreKey not in self.computationCoreDict:
                    raise ValueError("Computation core {} not found for feature {}.".format(coreKey, featureKey))

        self.distributedVariables = distributedVariables

    def clone(self):
        return ComputationActivationNetwork(
            featureDict=self.featureDict,
            computationCoreDict=self.computationCoreDict,
            baseMeasureCoreDict=self.baseMeasureCoreDict,
            canParamDict=self.canParamDict
        )

    def create_activation_cores(self, featureKeys=None):
        if featureKeys is None:
            featureKeys = list(self.featureDict.keys())
        activationCores = dict()
        for featureKey in featureKeys:
            if not "passive" in self.featureDict[featureKey].featureProperties:
                activationCores.update(self.featureDict[featureKey].create_activation_cores(
                    canParam=self.canParamDict[featureKey], coreType=self.coreType
                ))
        return activationCores

    def create_cores(self):
        return {**self.computationCoreDict,
                **self.create_activation_cores(),
                **self.baseMeasureCoreDict}

    def get_partition_function(self):
        return engine.contract(self.create_cores(), openColors=[], contractionMethod=self.contractionMethod)[:]

    def create_energyDict(self, cutoffWeight=100):
        """
        Not a feature method, since the features do not know their computationCores nor canParams
        """
        energyDict = dict()
        for featureKey in self.featureDict:  # Different treatment of weights and activation vectors for energy terms, specific to each feature
            affectedComCores = {coreKey: self.computationCoreDict[coreKey] for coreKey in
                                self.featureDict[featureKey].affectedComputationCores}
            if isinstance(self.featureDict[featureKey],
                          ft.SingleSoftFeature):  # Then contract with interpretation core and multiply with float canonical parameter
                energyDict[featureKey] = (self.canParamDict[featureKey],
                                          {**affectedComCores,
                                              "intCore_" + featureKey: bc.create_interpretation_vector(
                                                  color=self.featureDict[featureKey].featureColors[0],
                                                  coreType=self.coreType,
                                                  name="intCore_" + featureKey)})
            elif isinstance(self.featureDict[featureKey], ft.SoftPartitionFeature):
                energyDict[featureKey] = (1,
                                          {**affectedComCores,
                                           "canParamCore_" + featureKey: self.canParamDict[featureKey]})
            elif isinstance(self.featureDict[featureKey], ft.HardPartitionFeature):
                energyDict[featureKey] = (cutoffWeight,
                                          {**affectedComCores,
                                           "headCore_" + featureKey: self.canParamDict[featureKey]})
            elif isinstance(self.featureDict[featureKey], ft.EnergyDictFeature):
                energyDict.update(self.canParamDict[featureKey])

            else:
                raise ValueError("Unsupported feature type {}.".format(type(self.featureDict[featureKey])))
        energyDict["baseMeasure"] = (cutoffWeight, self.baseMeasureCoreDict)
        return energyDict

    def include_features(self, featureDict, canParamDict=dict(), computationCores=dict()):
        self.featureDict.update(featureDict)
        self.canParamDict.update(canParamDict)
        self.canParamDict.update({featureKey: self.featureDict[featureKey].find_neutral_canParam()
                                  for featureKey in featureDict if featureKey not in self.canParamDict})
        self.computationCoreDict.update(computationCores)

    def include(self, otherCANet):
        self.include_features(featureDict=otherCANet.featureDict, canParamDict=otherCANet.canParamDict,
                              computationCores=otherCANet.computationCoreDict)
