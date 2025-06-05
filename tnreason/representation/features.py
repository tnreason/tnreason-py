from tnreason import representation, engine

import math
import numpy as np

canCorePre = "_can"


class ComputedFeature:
    def __init__(self, featureColors, affectedComputationCores=[], shape=None, name=None):
        self.featureColors = featureColors
        self.affectedComputationCores = affectedComputationCores

        if name is None:
            self.name = "_".join(featureColors)
        else:
            self.name = str(name)

        if shape is None:
            self.shape = [2 for _ in self.featureColors]
        else:
            self.shape = shape

class PassiveFeature(ComputedFeature):
    featureType = "PassiveFeature"
#    def __init__(self, **feat):
#        super().__init__(featureColors=featureColors, affectedComputationCores=affectedComputationCores, shape=[])

    def create_activation_cores(self, **kwargs):
        return dict()

class SingleSoftFeature(ComputedFeature):
    featureType = "SingleSoftFeature"
    """
    One-dimensional (scalar) canonical parameter and mean parameter, which represents the exponentiation of the interpreted image.
    """

    def __init__(self, featureColor, interpretedImage=[0,1], **featSpec):
        super().__init__(featureColors=[featureColor], **featSpec)
        self.interpretationVector = representation.create_interpretation_vector(
                                    color=featureColor,
                                    interImage=interpretedImage)
        self.interpretedImage = interpretedImage
    def create_trivial_canParam(self, coreType=None):
        return 0

    def create_activation_cores(self, canParam, coreType=None):
        return {self.name + canCorePre + representation.suf.actCoreSuf: representation.create_activation_vector(
            color=self.featureColors[0],
            canParam=canParam,
            interImage=self.interpretedImage,
            coreType=coreType,
            name=self.name + canCorePre + representation.suf.actCoreSuf
        )}

    def create_activation_core(self, canParam, coreType=None):
        return representation.create_activation_vector(
            color=self.featureColors[0],
            canParam=canParam,
            interImage=self.interpretedImage,
            coreType=coreType,
            name=self.name + canCorePre + representation.suf.actCoreSuf
        )

    def compute_meanParam(self, environmentMean):
        return engine.contract({"envMean": environmentMean,
                                "intCore": self.interpretationVector
                                }, openColors=[])[:]

    def local_adjustment(self, environmentMean, meanParam, oldCanParam=None):
        """
        Adjusts the activation vector based on the indicator mean tensor.
        environmentMean: mean parameter, when canonical parameter
        """
        if self.interpretedImage == [0, 1]:
            assert meanParam <= 1 and meanParam >= 0
            if meanParam in [0, 1]:
                raise ValueError("Mean parameter {} needs to be a constraint.".format(meanParam))
            if environmentMean[0] == 0:
                # raise ValueError("Feature cannot be tuned!")
                return 0
            if environmentMean[1] == 0:
                # raise ValueError("Feature cannot be tuned!")
                return 0
            return oldCanParam + np.log(meanParam / (1 - meanParam) * (environmentMean[0] / environmentMean[1]))
        else:
            return newton_canonical_optimization(environmentMean, meanParam,
                                                 imageInterpretation=self.interpretedImage)

    def combine_canParams(self, canParamList):
        """
        Combine the canonical parameters such that the activation core contraction is reproduced
        """
        return sum(canParamList)

class SoftPartitionFeature(ComputedFeature):
    featureType = "SoftPartitionFeature"
    """
    Feature Colors are head colors of computation cores and the indices are indicating subsets of a partition.
    canParams and meanParams are tensors with the feature colors
    """

    def create_trivial_canParam(self, coreType=None):
        return 0 * representation.create_trivial_core(
            name=canCorePre, shape=self.shape,
            colors=self.featureColors, coreType=coreType)

    def create_activation_cores(self, canParam, coreType=None):
        return {self.name + canCorePre + representation.suf.actCoreSuf : representation.coordinatewise_transform(
            coreList=[canParam], rDrFunction=math.exp, outCoreType=coreType,
            outName=self.name + canCorePre + representation.suf.actCoreSuf,
        )}

    def create_activation_core(self, canParam, coreType=None):
        return representation.coordinatewise_transform(
            coreList=[canParam], rDrFunction=math.exp, outCoreType=coreType,
            outName=self.name + canCorePre + representation.suf.actCoreSuf,
        )

    def compute_meanParam(self, environmentMean):
        """
        For partition features the mean parameter is the environment mean.
        """
        return environmentMean

    def local_adjustment(self, environmentMean, meanParam, oldCanParam, coreType=None):
        """
        Do the IPF adjustment.
        """
        return representation.coordinatewise_transform(
            coreList=[oldCanParam, meanParam, environmentMean],
            rDrFunction=lambda old, soll, ist: old + np.log(soll / ist),
            outCoreType=coreType, outName=self.name + canCorePre + representation.suf.actCoreSuf
        )

    def combine_canParams(self, canParamList):
        """
        Combine the canonical parameters such that the activation core contraction is reproduced
        """
        if len(canParamList) == 1: ## Avoid problem with single core -> Would try 0 + canparamList[0]?
            return canParamList[0]
        return sum(canParamList)

class HardPartitionFeature(ComputedFeature):
    featureType = "HardPartitionFeature"
    """
    Activation cores are boolean tensors -> Interpretation as boolean base measure of the family<
    ! canParam interpreted as activation vector
    """

    def create_trivial_canParam(self, coreType=None):
        return representation.create_trivial_core(
            name=canCorePre, shape=self.shape,
            colors=self.featureColors, coreType=coreType)

    def create_activation_cores(self, canParam, coreType=None):
        return {self.name + canCorePre + representation.suf.actCoreSuf: canParam}

    def create_activation_core(self, canParam, coreType=None):
        return canParam

    def compute_meanParam(self, environmentMean=None):
        return self.local_adjustment(environmentMean=environmentMean)

    def local_adjustment(self, environmentMean=None, meanParam=None, oldCanParam=None, coreType=None):
        """
        Return subset encoding of the support intersection within the interpretation image
        """
        newCanparam = representation.create_vanishing_core(
            colors=self.featureColors,
            shape=self.shape,
            coreType=coreType)
        for posTuple in np.ndindex(*newCanparam.shape):
            posDict = {color: posTuple[i] for i, color in enumerate(newCanparam.colors)}
            keep = True
            if environmentMean is not None:
                if environmentMean[posDict] == 0:
                    keep = False
            if meanParam is not None:
                if meanParam[posDict] == 0:
                    keep = False
            if oldCanParam is not None:
                if oldCanParam[posDict] == 0:
                    keep = False
            if keep:
                newCanparam[posDict] = 1
        return newCanparam

    def combine_canParams(self, canParamList):
        """
        Combine the canonical parameters such that the activation core contraction is reproduced
        """
        return engine.contract(
            {"can"+str(i):canCore for i, canCore in enumerate(canParamList)},
            openColors=self.featureColors
        )


class ComputationActivationNetwork(engine.EngineUser):
    def __init__(self, featureDict, computationCoreDict=dict(), baseMeasureCoreDict=dict(), canParamDict=dict(),
                 **engineSpec):
        """
        * featureSpecDict: {featureKey : ExpDistFeature for all features}
        * interpretationDict: {featureKey : List of values} for all features
        """

        super().__init__(**engineSpec)

        self.featureDict = featureDict
        self.computationCoreDict = computationCoreDict
        self.baseMeasureCoreDict = baseMeasureCoreDict

        self.canParamDict = canParamDict

        for featureKey in self.featureDict:
            if featureKey not in self.canParamDict and type(self.featureDict[featureKey]) != PassiveFeature:
                self.canParamDict[featureKey] = self.featureDict[featureKey].create_trivial_canParam()
            for coreKey in self.featureDict[featureKey].affectedComputationCores:
                if coreKey not in self.computationCoreDict:
                    raise ValueError("Computation core {} not found for feature {}.".format(coreKey, featureKey))
    def create_activation_cores(self, featureKeys=None):
        for featureKey in self.featureDict:
            self.featureDict[featureKey].name = featureKey

        if featureKeys is None:
            featureKeys = list(self.featureDict.keys())
        activationCores = dict()
        for featureKey in featureKeys:
            activationCores.update(self.featureDict[featureKey].create_activation_cores(
                canParam=self.canParamDict[featureKey], coreType=self.coreType
            ))
        # # return activationCores
        # singleCreated = {featureKey + representation.suf.actCoreSuf: self.featureDict[featureKey].create_activation_core(
        #    canParam=self.canParamDict[featureKey],
        #    coreType=self.coreType) for
        #    featureKey in featureKeys if type(self.featureDict[featureKey]) != PassiveFeature}
        return activationCores


    def create_cores(self):
        return {**self.computationCoreDict,
                **self.create_activation_cores(),
                **self.baseMeasureCoreDict}

    def create_energyDict(self, cutoffWeight=100):
        """
        Not a feature method, since the features do not know their computationCores nor canParams
        """
        energyDict = dict()
        for featureKey in self.featureDict:  # Different treatment of weights and activation vectors for energy terms, specific to each feature
            affectedComCores = {coreKey: self.computationCoreDict[coreKey] for coreKey in
                                self.featureDict[featureKey].affectedComputationCores}
            if isinstance(self.featureDict[featureKey], SingleSoftFeature):
                energyDict[featureKey] = (self.canParamDict[featureKey],
                                          {**affectedComCores,
                                              "intCore_" + featureKey: representation.create_interpretation_vector(
                                                  color=self.featureDict[featureKey].featureColors[0],
                                                  coreType=self.coreType,
                                                  name="intCore_" + featureKey)})
            elif isinstance(self.featureDict[featureKey], SoftPartitionFeature):
                energyDict[featureKey] = (1,
                                          {**affectedComCores,
                                           "canParamCore_" + featureKey: self.canParamDict[featureKey]})
            elif isinstance(self.featureDict[featureKey], HardPartitionFeature):
                energyDict[featureKey] = (cutoffWeight,
                                          {**affectedComCores,
                                           "headCore_" + featureKey: self.canParamDict[featureKey]})
            else:
                raise ValueError("Unsupported feature type {}.".format(type(self.featureDict[featureKey])))
        energyDict["baseMeasure"] = (cutoffWeight, self.baseMeasureCoreDict)
        return energyDict

    def include_features(self, featureDict, canParamDict=dict(), computationCores=dict()):
        self.featureDict.update(featureDict)
        self.canParamDict.update(canParamDict)
        self.canParamDict.update({featureKey: self.featureDict[featureKey].create_trivial_canParam()
                                  for featureKey in featureDict if featureKey not in self.canParamDict})
        self.computationCoreDict.update(computationCores)


def calculate_single_canonical(indicatorMeanVector, meanParameter, imageInterpretation=[0, 1]):
    if len(imageInterpretation) == 1:
        return 0
    elif imageInterpretation == [0, 1]:
        if meanParameter in [0, 1]:
            raise ValueError("Mean parameter {} needs to be a constraint.".format(meanParameter))
        if indicatorMeanVector[0] == 0:
            raise ValueError("Indicator mean vector needs to be treated a constraint.")
        if indicatorMeanVector[1] == 0:
            raise ValueError("Indicator mean vector needs to be treated a constraint.")
        return np.log(meanParameter / (1 - meanParameter) * (indicatorMeanVector[0] / indicatorMeanVector[1]))
    else:
        return newton_canonical_optimization(indicatorMeanVector, meanParameter,
                                             imageInterpretation=imageInterpretation)


def newton_canonical_optimization(indicatorMeanVector, meanParam, imageInterpretation, startCanParam=0,
                                  maxIter=100, precision=1e-4, dumpFactor=0.8, verbose=True):
    """
    For single feature optimization, in case of imageInterpretation different from [0,1]
    """

    currentCanParam = startCanParam

    stopOptimization = False
    i = 0
    while not stopOptimization:
        i += 1
        if i > maxIter:
            stopOptimization = True
        newCanParam = newton_step_single(indicatorMeanVector, meanParam, currentCanParam,
                                         imageInterpretation=imageInterpretation)
        if abs(currentCanParam - newCanParam) < precision:
            stopOptimization = True
        currentCanParam = (1 - dumpFactor) * currentCanParam + dumpFactor * newCanParam

        if verbose:
            print("Newton Iteration {} updated parameter to {}.".format(i, currentCanParam))
    return currentCanParam


def newton_step_single(indicatorMeanVector, meanParameter, currCanParameter, imageInterpretation):
    color = indicatorMeanVector.colors[0]
    funvalue = engine.contract({"indicatorMean": indicatorMeanVector,
                                "currentCanCore": representation.create_activation_vector(
                                    color=color, canParam=currCanParameter, interImage=imageInterpretation),
                                "funTransform": representation.coordinatewise_transform(
                                    [representation.create_interpretation_vector(color=indicatorMeanVector.colors[0],
                                                                                 interImage=imageInterpretation)],
                                    rDrFunction=lambda x: (x - meanParameter)
                                )
                                }, openColors=[]
                               )[:]
    derValue = engine.contract({"indicatorMean": indicatorMeanVector,
                                "currentCanCore": representation.create_activation_vector(
                                    color=color, canParam=currCanParameter, interImage=imageInterpretation),
                                "funTransform": representation.coordinatewise_transform(
                                    [representation.create_interpretation_vector(color=indicatorMeanVector.colors[0],
                                                                                 interImage=imageInterpretation)],
                                    rDrFunction=lambda x: x * (x - meanParameter)
                                )
                                }, openColors=[]
                               )[:]
    return currCanParameter - funvalue / derValue if derValue != 0 else currCanParameter
