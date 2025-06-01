from tnreason import representation, engine

import math
import numpy as np

canCorePre = "_can"

class ComputedFeature:
    def __init__(self, featureColors, affectedComputationCores=[], interpretationDict=dict(), name=None):
        self.featureColors = featureColors
        self.affectedComputationCores = affectedComputationCores

        self.interpretationDict = interpretationDict
        for featureColor in featureColors:
            if featureColor not in interpretationDict:
                self.interpretationDict[featureColor] = [0, 1]

        self.name = str(name)


class SingleSoftFeature(ComputedFeature):
    """
    One-dimensional (scalar) canonical parameter and mean parameter, which represents the exponentiation of the interpreted image.
    """

    def __init__(self, featureColor, **featSpec):
        super().__init__(featureColors=[featureColor], **featSpec)

    def create_trivial_canParam(self, coreType=None):
        return 0

    def create_activation_core(self, canParam, coreType=None):
        return representation.create_activation_vector(
            color=self.featureColors[0],
            canParam=canParam,
            interImage=self.interpretationDict[self.featureColors[0]],
            coreType=coreType,
            name=self.featureColors[0] + representation.suf.actCoreSuf
        )

    def compute_meanParam(self, environmentMean):
        return engine.contract({"envMean": environmentMean,
                                "intCore": representation.create_interpretation_vector(
                                    color=self.featureColors[0],
                                    interImage=self.interpretationDict[self.featureColors[0]])
                                }, openColors=[])[:]

    def local_adjustment(self, environmentMean, meanParam, oldCanParam=None):
        """
        Adjusts the activation vector based on the indicator mean tensor.
        environmentMean: mean parameter, when canonical parameter
        """
        if self.interpretationDict[self.featureColors[0]] == [0, 1]:
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
                                                 imageInterpretation=self.interpretationDict[self.featureColors[0]])


class SoftPartitionFeature(ComputedFeature):
    """
    Feature Colors are head colors of computation cores and the indices are indicating subsets of a partition.
    canParams and meanParams are tensors with the feature colors
    """

    def create_trivial_canParam(self, coreType=None):
        return 0 * representation.create_trivial_core(
            name=canCorePre, shape=[len(self.interpretationDict[featureColor]) for featureColor in self.featureColors],
            colors=self.featureColors, coreType=coreType)

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

    def local_adjustment(self, environmentMean, meanParam, oldCanParam=None, coreType=None):
        """
        Do the IPF adjustment.
        """
        return representation.coordinatewise_transform(
            coreList=[oldCanParam, meanParam, environmentMean],
            rDrFunction=lambda old, soll, ist: old + np.log(soll / ist),
            outCoreType=coreType, outName=self.name + canCorePre + representation.suf.actCoreSuf
        )


class HardPartitionFeature(ComputedFeature):
    """
    Activation cores are boolean tensors -> Interpretation as boolean base measure of the family<
    ! canParam interpreted as activation vector
    """

    def create_trivial_canParam(self, coreType=None):
        return representation.create_trivial_core(
            name=canCorePre, shape=[len(self.interpretationDict[featureColor]) for featureColor in self.featureColors],
            colors=self.featureColors, coreType=coreType)

    def create_activation_core(self, canParam, coreType=None):
        return canParam

    def local_adjustment(self, environmentMean=None, meanParam=None, oldCanParam=None, coreType=None):
        """
        Return subset encoding of the support intersection within the interpretation image
        """
        newCanparam = representation.create_vanishing_core(
            colors=self.featureColors,
            shape=[len(self.interpretationDict[featureColor]) for featureColor in self.featureColors],
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


class ComputationActivationNetwork(engine.EngineUser):
    def __init__(self, featureDict, computationCoreDict=dict(), baseMeasureCoreDict=dict(), canParamDict=dict(),
                 # meanParamDict=dict(),
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
            if featureKey not in self.canParamDict:
                self.canParamDict[featureKey] = self.featureDict[featureKey].create_trivial_canParam()

    def create_activation_cores(self, featureKeys=None):
        if featureKeys is None:
            featureKeys = list(self.featureDict.keys())
        return {featureKey + representation.suf.actCoreSuf: self.featureDict[featureKey].create_activation_core(
            canParam=self.canParamDict[featureKey],
            coreType=self.coreType) for
            featureKey in featureKeys}

    def create_cores(self):
        return {**self.computationCoreDict,
                **self.create_activation_cores(),
                **self.baseMeasureCoreDict}

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
