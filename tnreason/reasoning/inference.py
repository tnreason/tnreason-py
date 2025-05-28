from tnreason import representation, engine

import numpy as np

canCorePre = "_can"  # Suffix for canonical parameter cores, to distinguish from the exponentiated ones being propert activation cores


class InferenceBase(engine.EngineUser):

    def __init__(self, computationCores, singleFeatureDict=dict(), partitionFeatureDict=dict(), interpretationDict=None,
                 canparamDict=None, meanparamDict=None, **engineSpec):
        super().__init__(**engineSpec)

        self.computationCores = computationCores  # dictionary of tensor cores preparing the feature variables Y_l

        self.singleFeatureDict = singleFeatureDict  # to each single feature stores a list of required computation cores
        self.partitionFeatureDict = partitionFeatureDict

        if interpretationDict is None:
            self.interpretationDict = {key: [0, 1] for key in self.singleFeatureDict}
        else:
            self.interpretationDict = interpretationDict

        if canparamDict is None:
            self.canParamDict = {**{key: float(0) for key in self.singleFeatureDict.keys()},
                                 **{key: 0 * representation.create_trivial_core(
                                     name=key + canCorePre + representation.suf.actCoreSuf,
                                     shape=[self.dimensionDict[key]],
                                     colors=[key],
                                     coreType=self.coreType) for key in self.partitionFeatureDict}
                                 }  # default: 0
        else:
            self.canParamDict = canparamDict

        if meanparamDict is None:
            self.meanParamDict = {**{key: float(0.5) for key in self.singleFeatureDict.keys()},
                                  **{key: 0.5 * representation.create_trivial_core(
                                      name=key + canCorePre + representation.suf.actCoreSuf,
                                      shape=[self.dimensionDict[key]],
                                      colors=[key],
                                      coreType=self.coreType) for key in self.partitionFeatureDict}
                                  }  # default: 0.5
        else:
            self.meanParamDict = meanparamDict

    def create_activation_cores(self, featureColors=None):
        if featureColors is None:
            featureColors = list(self.singleFeatureDict.keys()) + list(self.partitionFeatureDict.keys())
        return {**{featureColor + representation.suf.actCoreSuf: representation.create_activation_vector(
            featureColor, canParam=self.canParamDict[featureColor], supportConstraint=[], coreType=self.coreType,
            interImage=self.interpretationDict[featureColor], name=featureColor + representation.suf.actCoreSuf
        )
            for featureColor in featureColors if featureColor in self.singleFeatureDict
        },
                **{featureColor + representation.suf.actCoreSuf: representation.create_partition_activation_vector(
                    featureColor, canParamCore=self.canParamDict[featureColor],
                    name=featureColor + representation.suf.actCoreSuf
                )
                    for featureColor in featureColors if featureColor in self.partitionFeatureDict
                }}


class ForwardContractor(InferenceBase):
    """
    Calculates mean parameters by direct contraction
    """

    def calculate_mean_parameters(self, featureColors=None):
        if featureColors is None:
            featureColors = self.singleFeatureDict.keys()

        for featureColor in featureColors:
            if featureColor in self.singleFeatureDict:
                self.meanParamDict[featureColor] = self.calculate_single_mean(featureColor)
            elif featureColor in self.partitionFeatureDict:
                self.meanParamDict[featureColor] = self.calculate_indicator_mean(featureColor)

    def calculate_single_mean(self, featureColor):
        """
        Calculates the mean for a single feature.
        """
        return engine.contract(
            coreDict={**self.computationCores,
                      **self.create_activation_cores(),
                      "interpretationCore": representation.create_interpretation_vector(featureColor,
                                                                                        coreType=self.coreType,
                                                                                        interImage=
                                                                                        self.interpretationDict[
                                                                                            featureColor])},
            openColors=[]
        )[:]

    def calculate_indicator_mean(self, featureColor):
        """
        Calculates the mean for the partition statistics by the image element indicators
        to the feature specified by computation color featureColor.
        """
        return engine.contract(
            coreDict={**self.computationCores,
                      **self.create_activation_cores()},
            openColors=[featureColor]
        )


class BackwardAlternator(InferenceBase):

    def __init__(self, computationCores, singleFeatureDict, partitionFeatureDict, interpretationDict=None,
                 canparamDict=None,
                 meanparamDict=None,
                 forwardInferer=None, **engineSpec):
        super().__init__(computationCores, singleFeatureDict=singleFeatureDict,
                         partitionFeatureDict=partitionFeatureDict, interpretationDict=interpretationDict,
                         canparamDict=canparamDict, meanparamDict=meanparamDict,
                         **engineSpec)
        if forwardInferer is None:
            """
            By default use the contraction based forward inferer
            """
            self.forwardInferer = ForwardContractor(computationCores=computationCores,
                                                    singleFeatureDict=singleFeatureDict,
                                                    interpretationDict=interpretationDict, canparamDict=canparamDict,
                                                    meanparamDict=meanparamDict, **engineSpec)
        elif not isinstance(forwardInferer, ForwardContractor):
            raise ValueError("forwardInferer needs to be an instance of ForwardContractor.")
        else:
            self.forwardInferer = forwardInferer

    def update_partition_feature(self, featureColor):
        """
        Updates the canonical parameter vector of a partition feature
        """
        if featureColor not in self.partitionFeatureDict:
            raise ValueError("Feature color {} not in partitionFeatureDict.".format(featureColor))

        indicatorMean = self.forwardInferer.calculate_indicator_mean(featureColor)
        self.canParamDict[featureColor] = representation.coordinatewise_transform(
            coreList=[self.canParamDict[featureColor], self.meanParamDict[featureColor], indicatorMean],
            rDrFunction=lambda old, soll, ist: old + np.log(soll / ist),
            outCoreType=self.coreType, outName=featureColor + canCorePre + representation.suf.actCoreSuf
        )
        self.forwardInferer.canParamDict[featureColor] = self.canParamDict[featureColor]

    def update_single_feature(self, featureColor):
        if featureColor not in self.singleFeatureDict:
            raise ValueError("Feature color {} not in singleFeatureDict.".format(featureColor))

        self.forwardInferer.canParamDict[featureColor] = 0
        indicatorMean = self.forwardInferer.calculate_indicator_mean(featureColor)
        canParamSolution = calculate_optimal_canonical(
            indicatorMeanVector=indicatorMean,
            meanParameter=self.meanParamDict[featureColor],
            imageInterpretation=self.interpretationDict[featureColor]
        )
        self.canParamDict[featureColor] = canParamSolution
        self.forwardInferer.canParamDict[featureColor] = canParamSolution

    def alternating_updates(self, featureColors=None, sweepNum=10):
        if featureColors is None:
            featureColors = list(self.singleFeatureDict.keys()) + list(self.partitionFeatureDict.keys())

        for _ in range(sweepNum):
            for featureColor in featureColors:
                if featureColor in self.partitionFeatureDict:
                    self.update_partition_feature(featureColor)
                else:
                    self.update_single_feature(featureColor)


def calculate_optimal_canonical(indicatorMeanVector, meanParameter, imageInterpretation=[0, 1]):
    if imageInterpretation == [0, 1]:
        if meanParameter in [0, 1]:
            raise ValueError("Mean parameter {} needs to be a constraint.".format(meanParameter))
        if indicatorMeanVector[0] == 0:
            raise ValueError("Indicator mean vector needs to be treated a constraint.")
        if indicatorMeanVector[1] == 0:
            raise ValueError("Indicator mean vector needs to be treated a constraint.")
        return np.log(meanParameter / (1 - meanParameter) * (indicatorMeanVector[0] / indicatorMeanVector[1]))
    else:
        ValueError("Image interpretation {} not supported.".format(imageInterpretation))
