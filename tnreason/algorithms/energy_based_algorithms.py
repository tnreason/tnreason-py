from tnreason import engine

from tnreason.algorithms import sampling_handling as sh

import numpy as np

## Energy-based
gibbsMethodString = "gibbsSample"
meanFieldMethodString = "meanFieldSample"
energyMaximumMethodString = "exactEnergyMax"
energyOptimizationMethods = [gibbsMethodString, meanFieldMethodString, energyMaximumMethodString]


def optimize_energy(energyDict=[], method=gibbsMethodString, **specDict):
    """
    Mixes two steps! Approximation (EnergyMeanFieldApproximator) and Sample Drawing (EnergyGibbsSampleCore)
    """
    if method == gibbsMethodString:
        sampler = EnergyGibbsSampleCore(energyDict=energyDict, **specDict)
        sampler.draw_sample()
        return sampler.sample
    elif method == meanFieldMethodString:
        approximator = NaiveMeanFieldApproximator(energyDict=energyDict, **specDict)
        approximator.anneal(
            approximationTemperatureList=specDict.get("approximationTemperatureList", [1 for i in range(10)]))

        #        return EnergyGibbsSampleCore(energyDict=approximator.get_energyDict(), **specDict)
        return approximator.draw_sample()
    elif method == energyMaximumMethodString:
        """
        Potential: Sparse Cores -> HUBO!
        """
        contracted = engine.create_trivial_core("contracted", [specDict.get("dimDict", dict())[color] for color in
                                                               specDict.get("colors", [])], specDict.get("colors", []),
                                                coreType=specDict.get("coreType", None))
        for energyKey in energyDict:
            contracted = contracted.sum_with(
                engine.contract(energyDict[energyKey][0], openColors=specDict.get("colors", []),
                                dimDict=specDict.get("dimDict", dict()),
                                method=specDict.get("contractionMethod", None)).multiply(
                    energyDict[energyKey][1]))
        return contracted.get_argmax()
    else:
        raise ValueError("Energy Optimization Method {} not implemented.".format(energyMaximumMethodString,
                                                                                 method))


class GenericMeanFieldApproximator(engine.EngineUser):
    def __init__(self, energyDict, colors=[], edgeColorDict=None, **engineSpec):
        """
        Edge Color Dict : Representing a HyperGraph, which Markov Network is used for approximation
        """
        super().__init__(**engineSpec)

        self.energyDict = energyDict
        self.colors = colors

        if edgeColorDict is None:
            self.edgeColorDict = {color: [color] for color in colors}
        else:
            self.edgeColorDict = edgeColorDict

        self.approxCores = {parKey: engine.create_trivial_core(parKey, [self.dimensionDict[color] for color in
                                                                        self.edgeColorDict[parKey]],
                                                               self.edgeColorDict[parKey],
                                                               coreType=self.coreType).multiply(
            1 / np.prod([self.dimensionDict[color] for color in self.edgeColorDict[parKey]])) for parKey in
            self.edgeColorDict}

    def update_core(self, approxCoreKey):

        restApproxCores = {key: self.approxCores[key] for key in self.approxCores if key != approxCoreKey}
        contracted = engine.sum_contract(
            weightedCoreDicts=energyDict_to_weightedCoresDicts(self.energyDict) +
                              [(-1, {key: self.approxCores[key].build_ln()}) for key in self.approxCores if
                               key != approxCoreKey],
            backCores=restApproxCores, openColors=self.edgeColorDict[approxCoreKey],
            coreType=self.coreType,
            contractionMethod=self.contractionMethod
        )
        ## Normate coordinatewise
        denominator = engine.contract(restApproxCores, openColors=self.edgeColorDict[approxCoreKey])
        update = engine.get_core(self.coreType)(values=None, colors=self.edgeColorDict[approxCoreKey],
                                                shape=[self.dimensionDict[color] for color in
                                                       self.edgeColorDict[approxCoreKey]])
        sum = 0
        for i in np.ndindex(*[self.dimensionDict[color] for color in self.edgeColorDict[approxCoreKey]]):
            posDict = {color: i[l] for l, color in enumerate(self.edgeColorDict[approxCoreKey])}
            coordinate = np.exp(contracted[posDict] / denominator[posDict])
            update[posDict] = coordinate
            sum += 1
        self.approxCores[approxCoreKey] = update.multiply(1 / sum)

    def get_energyDict(self):
        return [(1, {coreKey: self.approxCores[coreKey].build_ln()}) for coreKey in self.approxCores]


class NaiveMeanFieldApproximator(engine.EngineUser):
    def __init__(self, energyDict, colors=[], partionColorDict=None, **engineSpec):
        super().__init__(**engineSpec)

        self.colors = colors
        self.energyDict = energyDict

        self.affectionDict = create_affectionDict(energyDict, self.colors)

        if partionColorDict is None:
            self.partitionColorDict = {color: [color] for color in self.colors}
        else:
            self.partitionColorDict = partionColorDict

        # Only distinction to Gibbs: MeanCores instead of samples turned into cores
        self.meanCores = {parKey: engine.create_trivial_core(parKey, [self.dimensionDict[color] for color in
                                                                      self.partitionColorDict[parKey]],
                                                             self.partitionColorDict[parKey],
                                                             coreType=self.coreType).multiply(
            1 / np.prod([self.dimensionDict[color] for color in self.partitionColorDict[parKey]])) for parKey
            in self.partitionColorDict}

    def update_meanCore(self, upKey, temperature=1):

        oldMean = self.meanCores[upKey].clone()

        restMeanCores = {secKey: self.meanCores[secKey] for secKey in self.meanCores if secKey != upKey}
        affectedEnergyKeys = list(set().union(*[self.affectionDict[color] for color in self.partitionColorDict[upKey]]))

        contracted = engine.sum_contract(energyDict_to_weightedCoresDicts(self.energyDict, affectedEnergyKeys),
                                         backCores=restMeanCores, openColors=self.partitionColorDict[upKey],
                                         dimensionDict=self.dimensionDict, contractionMethod=self.contractionMethod,
                                         coreType=self.coreType)

        self.meanCores[upKey] = contracted.multiply(1 / temperature).exponentiate().normalize()

        angle = engine.contract({"old": oldMean, "new": self.meanCores[upKey]}, openColors=[],
                                method=self.contractionMethod)
        return angle.values

    def anneal(self, approximationTemperatureList):
        angles = np.empty(shape=(len(approximationTemperatureList), len(self.partitionColorDict)))
        for i, temperature in enumerate(approximationTemperatureList):
            for j, upKey in enumerate(self.partitionColorDict):
                angles[i, j] = self.update_meanCore(upKey, temperature=temperature)
        return angles

    def draw_sample(self):
        """
        Draws a sample from the approximating independent distribution
        -> Better to use EnergyGibbs for that!
        """
        sample = {}
        for coreKey in self.meanCores:
            sample.update(self.meanCores[coreKey].draw_sample(temperature=1))
        return sample

    def get_energyDict(self):
        return [(1, {coreKey: self.meanCores[coreKey].build_ln()}) for coreKey in self.meanCores]


class EnergyGibbsSampleCore(sh.SampleCoreBase):
    def __init__(self, energyDict, **specDict):
        super().__init__(**specDict)

        self.energyDict = energyDict
        self.affectionDict = create_affectionDict(energyDict, self.colors)
        self.temperatureList = specDict.get("temperatureList", [1])
        self.partitionColorDict = specDict.get("partitionColorDict", {color: [color] for color in self.colors})

    def draw_sample(self, startAssignment=dict()):
        self.sample = startAssignment
        for i, temperature in enumerate(self.temperatureList):
            for j, upKey in enumerate(self.partitionColorDict):
                self.sample_colors(self.partitionColorDict[upKey], temperature=temperature)
        return self.sample

    def sample_colors(self, colors, temperature=1):
        energy = self.calculate_energy(colors)
        self.sample.update(energy.draw_sample(asEnergy=True, temperature=temperature))

    def calculate_energy(self, upColors):
        affectedEnergyKeys = list(set().union(*[self.affectionDict[color] for color in upColors]))
        sampleCores = {
            color + "_sampleCore": engine.create_basis_core(color + "_sampleCore", [self.dimensionDict[color]], [color],
                                                            (self.sample[color]), coreType=self.coreType) for
            color in self.sample if color not in upColors}

        return engine.sum_contract(energyDict_to_weightedCoresDicts(self.energyDict, affectedEnergyKeys),
                                   backCores=sampleCores, openColors=upColors, dimensionDict=self.dimensionDict,
                                   contractionMethod=self.contractionMethod, coreType=self.coreType)


def create_affectionDict(energyDict, colors):
    return {color: [energyKey for energyKey in energyDict if
                    any([color in energyDict[energyKey][0][coreKey].colors for coreKey in energyDict[energyKey][0]])]
            for color in colors}


def energyDict_to_weightedCoresDicts(energyDict, useKeys=None):
    if useKeys is None:
        useKeys = list(energyDict.keys())
    return [(energyDict[useKey][1], energyDict[useKey][0]) for useKey in useKeys]
