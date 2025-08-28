import tnreason.representation.coordinate_calculus
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
    """
    Feature without active part, i.e. trivial activation core.
    Canonical Parameter and mean parameter are always None
    Serves as a selector of computation cores.
    """
    featureProperties = ["passive"]

    def create_activation_cores(self, **kwargs):
        # Avoid that usage in application, since not necessary in contractions
        return engine.create_from_slice_iterator(shape=self.shape, colors=self.featureColors, sliceIterator=[(1, {})])

    def combine_canParams(self, canParamList):
        return None

    def find_neutral_canParam(self, coreType=None):
        return None

    def local_adjustment(self, **kwargs):
        return None

    def compute_meanParam(self, **kwargs):
        return None


class SingleHybridFeature(ComputedFeature):
    featureProperties = []
    """
    Scalar canonical parameter, if boolean then hard and if float or int then soft
    """

    def __init__(self, featureColor, interpretedImage=[0, 1], **featSpec):
        super().__init__(featureColors=[featureColor], **featSpec)
        self.interpretationVector = representation.create_interpretation_vector(
            color=featureColor,
            interImage=interpretedImage)
        self.interpretedImage = interpretedImage

    def find_neutral_canParam(self, coreType=None):
        return 0

    def create_activation_cores(self, canParam, coreType=None):
        if isinstance(canParam, bool):  ## Hard activation
            return {self.name + canCorePre + representation.suf.actCoreSuf: representation.create_basis_core(
                name=self.name + canCorePre + representation.suf.actCoreSuf,
                shape=self.shape, colors=self.featureColors, numberTuple=(int(canParam),),
                coreType=coreType
            )}
        elif isinstance(canParam, float) or isinstance(canParam, int):  ## Soft activation
            return {self.name + canCorePre + representation.suf.actCoreSuf: representation.coordinatewise_transform(
                [self.interpretationVector], rDrFunction=lambda x: math.exp(canParam * x), outCoreType=coreType,
                outName=self.name + canCorePre + representation.suf.actCoreSuf
            )}
        else:
            raise ValueError("Canonical parameter must be a float, int or bool, but is {}".format(type(canParam)))

    def compute_meanParam(self, environmentMean):
        return engine.contract({"envMean": environmentMean,
                                "intCore": self.interpretationVector
                                }, openColors=[])[:]

    def local_adjustment(self, environmentMean, meanParam, oldCanParam=None, cutoffWeight=10):
        """
        Adjusts the activation vector based on the indicator mean tensor.
        environmentMean: mean parameter, when canonical parameter
        """
        if self.interpretedImage == [0, 1]:
            assert meanParam <= 1 and meanParam >= 0
            if meanParam in [0, 1]:
                return bool(meanParam)
            if environmentMean[0] == 0:
                assert meanParam != 0
                return True
            if environmentMean[1] == 0:
                assert meanParam != 1
                return False
            return oldCanParam + np.log(meanParam / (1 - meanParam) * (environmentMean[0] / environmentMean[1]))
        else:
            return newton_canonical_optimization(environmentMean, meanParam,
                                                 interpretationVector=self.interpretationVector)


class SingleSoftFeature(ComputedFeature):
    featureProperties = []
    """
    One-dimensional (scalar) canonical parameter and mean parameter, which represents the exponentiation of the interpreted image.
    """

    def __init__(self, featureColor, interpretedImage=[0, 1], **featSpec):
        super().__init__(featureColors=[featureColor], **featSpec)
        self.interpretationVector = representation.create_interpretation_vector(
            color=featureColor,
            interImage=interpretedImage)
        self.interpretedImage = interpretedImage

    def find_neutral_canParam(self, coreType=None):
        return 0

    def create_activation_cores(self, canParam, coreType=None):
        return {self.name + canCorePre + representation.suf.actCoreSuf: representation.coordinatewise_transform(
            [self.interpretationVector], rDrFunction=lambda x: math.exp(canParam * x), outCoreType=coreType,
            outName=self.name + canCorePre + representation.suf.actCoreSuf
        )}

    def compute_meanParam(self, environmentMean):
        return engine.contract({"envMean": environmentMean,
                                "intCore": self.interpretationVector
                                }, openColors=[])[:]

    def local_adjustment(self, environmentMean, meanParam, oldCanParam=None, cutoffWeight=10):
        """
        Adjusts the activation vector based on the indicator mean tensor.
        environmentMean: mean parameter, when canonical parameter
        """
        if self.interpretedImage == [0, 1]:
            assert meanParam <= 1 and meanParam >= 0
            if meanParam in [0, 1]:
                raise ValueError("Mean parameter {} needs to be a constraint.".format(meanParam))
            if environmentMean[0] == 0:
                # Then a hard constraint, approximated by the cutoffWeight
                return cutoffWeight
            if environmentMean[1] == 0:
                # raise ValueError("Feature cannot be tuned!")
                return - cutoffWeight
            return oldCanParam + np.log(meanParam / (1 - meanParam) * (environmentMean[0] / environmentMean[1]))
        else:
            return newton_canonical_optimization(environmentMean, meanParam,
                                                 interpretationVector=self.interpretationVector)

    def combine_canParams(self, canParamList):
        """
        Combine the canonical parameters such that the activation core contraction is reproduced
        """
        return sum(canParamList)


class SoftPartitionFeature(ComputedFeature):
    featureProperties = []
    """
    Feature Colors are head colors of computation cores and the indices are indicating subsets of a partition.
    canParams and meanParams are tensors with the feature colors
    """

    def find_neutral_canParam(self, coreType=None):
        """
        Neutral canParams are those, where the contraction of the corresponding activation core with the affected computation cores is the trivial ones tensor.
        """
        return representation.create_vanishing_core(colors=self.featureColors, shape=self.shape,
                                                    name=self.name + canCorePre + representation.suf.actCoreSuf,
                                                    coreType=coreType)

    def create_activation_cores(self, canParam, coreType=None):
        return {self.name + canCorePre + representation.suf.actCoreSuf: representation.coordinatewise_transform(
            coreList=[canParam], rDrFunction=math.exp, outCoreType=coreType,
            outName=self.name + canCorePre + representation.suf.actCoreSuf,
        )}

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
        if len(canParamList) == 1:  ## Avoid problem with single core -> Would try 0 + canparamList[0]?
            return canParamList[0]
        return sum(canParamList)


class HardPartitionFeature(ComputedFeature):
    featureProperties = ["hard"]
    """
    Activation cores are boolean tensors -> Interpretation as boolean base measure of the family
    ! canParam interpreted as activation vector, without coordinatewise exponentiation (i.e. canParam is the limit of normed soft activation cores)
    """

    def find_neutral_canParam(self, coreType=None):
        return tnreason.representation.coordinate_calculus.create_trivial_core(
            name=canCorePre, shape=self.shape,
            colors=self.featureColors, coreType=coreType)

    def create_activation_cores(self, canParam, coreType=None):
        return {self.name + canCorePre + representation.suf.actCoreSuf: canParam}

    def compute_meanParam(self, environmentMean=None):
        return self.local_adjustment(environmentMean=environmentMean)

    def local_adjustment(self, environmentMean=None, meanParam=None, oldCanParam=None, coreType=None):
        """
        Return subset encoding of the support intersection within the interpretation image
        """
        sliceList = []
        for posTuple in np.ndindex(*self.shape):
            posDict = {color: posTuple[i] for i, color in enumerate(self.featureColors)}
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
                sliceList.append((1, posDict))
        return engine.create_from_slice_iterator(shape=self.shape, colors=self.featureColors,
                                                 sliceIterator=sliceList,
                                                 coreType=coreType,
                                                 name=self.name + canCorePre + representation.suf.actCoreSuf)

    def combine_canParams(self, canParamList):
        """
        Combine the canonical parameters such that the activation core contraction is reproduced
        """
        return engine.contract(
            {"can" + str(i): canCore for i, canCore in enumerate(canParamList)},
            openColors=self.featureColors
        )

    def activation_sum(self, canParam):
        return engine.contract({"can": canParam}, openColors=[])[:]
class EnergyDictFeature(ComputedFeature):
    """
    Dummy Feature with no active part.
    CA Network uses this to directly add the canonical parameter (which is a dict of weighted tensor networks) into the energy dict
    """
    featureProperties = ["passive"]

    def create_activation_cores(self, canParam, coreType=None):
        raise ValueError("Energy Dict Feature cannot create activation cores efficiently!")


class TNFeature(ComputedFeature):
    """
    Those with tensor networks as activation core.
    MeanParam computation and local adjustment are treated as in SoftPartitionFeature and produce single tensors.
    """
    featureProperties = []

    def create_activation_cores(self, canParam, coreType=None):
        return canParam

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


# def calculate_single_canonical(indicatorMeanVector, meanParameter, imageInterpretation=[0, 1]):
#     if len(imageInterpretation) == 1:
#         return 0
#     elif imageInterpretation == [0, 1]:
#         if meanParameter in [0, 1]:
#             raise ValueError("Mean parameter {} needs to be a constraint.".format(meanParameter))
#         if indicatorMeanVector[0] == 0:
#             raise ValueError("Indicator mean vector needs to be treated a constraint.")
#         if indicatorMeanVector[1] == 0:
#             raise ValueError("Indicator mean vector needs to be treated a constraint.")
#         return np.log(meanParameter / (1 - meanParameter) * (indicatorMeanVector[0] / indicatorMeanVector[1]))
#     else:
#         return newton_canonical_optimization(indicatorMeanVector, meanParameter,
#                                              imageInterpretation=imageInterpretation)


def newton_canonical_optimization(indicatorMeanVector, meanParam, interpretationVector, startCanParam=0,
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
                                         interpretationVector=interpretationVector)
        if abs(currentCanParam - newCanParam) < precision:
            stopOptimization = True
        currentCanParam = (1 - dumpFactor) * currentCanParam + dumpFactor * newCanParam

        if verbose:
            print("Newton Iteration {} updated parameter to {}.".format(i, currentCanParam))
    return currentCanParam


def newton_step_single(indicatorMeanVector, meanParameter, currCanParameter, interpretationVector):
    funvalue = engine.contract({"indicatorMean": indicatorMeanVector,
                                "currentActCore": representation.coordinatewise_transform(
                                    [interpretationVector], rDrFunction=lambda x: math.exp(currCanParameter * x),
                                    outName="currentActCore"
                                ),
                                "funTransform": representation.coordinatewise_transform(
                                    [interpretationVector],
                                    rDrFunction=lambda x: (x - meanParameter), outName="funTransForm"
                                )
                                }, openColors=[]
                               )[:]
    derValue = engine.contract({"indicatorMean": indicatorMeanVector,
                                "currentActCore": representation.coordinatewise_transform(
                                    [interpretationVector], rDrFunction=lambda x: math.exp(currCanParameter * x),
                                    outName="currentActCore"
                                ),
                                "funTransform": representation.coordinatewise_transform(
                                    [interpretationVector],
                                    rDrFunction=lambda x: x * (x - meanParameter)
                                )
                                }, openColors=[]
                               )[:]
    return currCanParameter - funvalue / derValue if derValue != 0 else currCanParameter
