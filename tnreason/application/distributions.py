from tnreason import representation
from tnreason import engine
from tnreason import reasoning

from tnreason.application import data_to_cores as dtc
from tnreason.application import formulas_to_cores as ftc
from tnreason.application import categoricals_to_cores as ctc
from tnreason.application import script_transform as st
from tnreason.application import storage as stor

import math
import numpy as np

probFormulasKey = "weightedFormulas"
logFormulasKey = "facts"
categoricalsKey = "categoricalConstraints"
evidenceKey = "evidence"

mnSoftFeatureSuffix = "_mSoft"
mnHardFeatureSuffix = "_mHard"


class DistributionBase(engine.EngineUser):
    """
    Distributions are creators of representation.ComputationActivationNetworks with additional features.
    Most important methods are:
        * create_caNetwork(): instantiates the corresponding ComputationActivationNetwork
        * get_partition_function(allAtoms): returning the partition function given the atomic variables of interest
    """

    def __init__(self, partitionFunction=None, distributedVariables=None, **engineSpec):
        super().__init__(**engineSpec)
        self.partitionFunction = partitionFunction
        self.distributedVariables = distributedVariables or []

    def create_cores(self):
        return self.create_caNetwork(hardOnly=False).create_cores()

    def get_partition_function(self, addDimDict={}):
        if self.partitionFunction is None:
            self.partitionFunction = engine.contract(self.create_cores(), openColors=[],
                                                     contractionMethod=self.contractionMethod)[:]
        return math.prod(
            [addDimDict[color] for color in addDimDict if color not in self.dimensionDict]) * self.partitionFunction

    def get_energy_dict(self, cutoffWeight=100):
        """
        Returns the energy dict (customized)
        """
        return self.create_caNetwork(hardOnly=False).create_energyDict(cutoffWeight=cutoffWeight)

    def is_normable(self):
        """
        For HybridKnowledgeBase: Decides whether the Knowledge Base is satisfiable, i.e. whether a model exists
        For CSP: Decides whether the CSP has a solution, i.e. whether a model exists
        Suffices to create the hard features, since only those restrict the support of the network (whereas the partition function needs all features)
        """
        return engine.contract(coreDict=self.create_cores(),
                               openColors=[])[:] > 0

    def as_core(self):
        """
        Represents the distribution as a single core
        """
        return engine.contract(self.create_cores(), openColors=self.distributedVariables,
                               contractionMethod=self.contractionMethod)


class ProposalDistribution(DistributionBase):

    def __init__(self, positivePhase, negativePhase, statisticCores, **distributionSpec):
        super().__init__(**distributionSpec)
        self.positivePhase = positivePhase
        self.negativePhase = negativePhase
        self.statisticCores = statisticCores

    def create_cores(self):
        raise ValueError("Cores of Proposal Distribution cannot be instantiated, only its energy can!")

    def create_caNetwork(self, hardOnly=None):
        """
        CANetwork can only produce energy dict!
        """
        correctionColorDict = {color: self.dimensionDict[color] for color in self.distributedVariables}
        return representation.ComputationActivationNetwork(
            featureDict={"energyDictProposal" : representation.EnergyDictFeature(featureColors=self.distributedVariables,
                                                         affectedComputationCores=[])},
            canParamDict={"energyDictProposal" :
                              {"pos": (1 / self.positivePhase.get_partition_function(correctionColorDict),
                                       {**self.statisticCores, **self.positivePhase.create_cores()}),
                               "neg": (-1 / self.negativePhase.get_partition_function(correctionColorDict),
                                       {**self.statisticCores, **self.negativePhase.create_cores()})}
                          },
            distributedVariables=self.distributedVariables
        )


class MarkovNetwork(DistributionBase):
    """
    Interprets a Tensor Network as a distribution
    """

    def __init__(self, coreDict, **distributionSpec):
        super().__init__(**distributionSpec)

        self.coreDict = coreDict
        self.dimDict = engine.get_dimDict(coreDict)

    def create_cores(self, hardOnly=False):
        return self.create_caNetwork(hardOnly=hardOnly).create_cores()

    def create_caNetwork(self, hardOnly=False):
        """
        Converts the Markov Network to a Computation Activation Network
        """
        featureDict = dict()
        canParamDict = dict()

        # Decide Hard Features
        for coreKey in self.coreDict:
            supportCore, needed = create_factor_hardCore(self.coreDict[coreKey], outCoreType=self.coreType,
                                                         outName=coreKey + mnHardFeatureSuffix)
            if needed:
                featureDict[coreKey + mnHardFeatureSuffix] = representation.HardPartitionFeature(
                    featureColors=supportCore.colors,
                    affectedComputationCores=[],
                )
                canParamDict[coreKey + mnHardFeatureSuffix] = supportCore
        if not hardOnly:
            for coreKey in self.coreDict:
                canParamCore, needed = create_factor_softCore(self.coreDict[coreKey], outCoreType=self.coreType,
                                                              outName=coreKey + mnSoftFeatureSuffix)
                if needed:
                    featureDict[coreKey + mnSoftFeatureSuffix] = representation.SoftPartitionFeature(
                        featureColors=self.coreDict[coreKey].colors,
                        affectedComputationCores=[],
                    )
                    canParamDict[coreKey + mnSoftFeatureSuffix] = canParamCore
        return representation.ComputationActivationNetwork(featureDict=featureDict,
                                                           canParamDict=canParamDict,
                                                           coreType=self.coreType,
                                                           computationCoreDict=dict(),
                                                           distributedVariables=self.distributedVariables
                                                           )


def get_empirical_distribution(sampleDf, atomColumns=None, interpretation="atomic", dimensionsDict=None):
    """
    Returns an empirical distributions as a MarkovNetwork
        * sampleDf: pd.DataFrame containing the samples defining the empirical distributions
        * atomKeys: List of columns of sampleDf to be recognized as atoms
        * interpretation: Specifies the interpretation of the entries of sampleDf
            - "atomic": Variables have dimension 2 and entries in [0,1] are the probability of the atom holding.
            - "categorical": Variables have dimension m specified in dimensionsDict and entries are the certain value of the variable in [m]
    """
    if atomColumns is not None:
        sampleDf = sampleDf[atomColumns]
    else:
        atomColumns = list(sampleDf.columns)
    if "value" not in sampleDf.columns:
        sampleDf["value"] = 1
    return MarkovNetwork(dtc.create_data_cores(sampleDf, atomKeys=atomColumns,
                                               interpretation=interpretation,
                                               dimensionsDict=dimensionsDict),
                         distributedVariables=[atomKey if atomKey.endswith(
                             representation.suf.disVarSuf) else atomKey + representation.suf.disVarSuf for atomKey in
                                               atomColumns],
                         partitionFunction=sampleDf["value"].sum())

class HybridLogicNetwork(DistributionBase):
    def __init__(self, formulaDict={}, canParamDict={}, **distributionSpec):
        super().__init__(**distributionSpec)
        self.formulaDict = formulaDict
        self.canParamDict = canParamDict

    def create_caNetwork(self):
        featureDict = {formulaKey : representation.SingleHybridFeature(
            featureColor = ftc.get_formula_headColor(self.formulaDict[formulaKey]),
            affectedComputationCores=list(
                ftc.create_formula_computation_cores(self.formulaDict[formulaKey]).keys()),
        ) for formulaKey in self.formulaDict}

        computationCoreDict = ftc.create_computation_cores_to_expressionDict(self.formulaDict, coreType=self.coreType)

        return representation.ComputationActivationNetwork(featureDict=featureDict,
                                                           canParamDict=self.canParamDict,
                                                           computationCoreDict=computationCoreDict,
                                                           #baseMeasureCoreDict={},
                                                           #distributedVariables=self.distributedVariables,
                                                           coreType=self.coreType)


class HybridKnowledgeBase(DistributionBase):
    """
    Inferable (by HybridInferer) Knowledge Base. Generalizes Markov Logic Network by further dedicated cores
    * dimensionDict: Dictionary of dimensions for in the formulas appearing categorical variables
    """

    def __init__(self, weightedFormulas={}, facts={}, categoricalConstraints={}, evidence={}, backCores={},
                 **distributionSpec):
        super().__init__(**distributionSpec)
        self.weightedFormulas = weightedFormulas
        #self.weightedFormulas = {key: weightedFormulas[key][:-1] + [float(weightedFormulas[key][-1])] for key in
        #                         weightedFormulas}
        self.facts = facts
        self.categoricalConstraints = categoricalConstraints
        self.evidence = evidence

        ## Option to add arbitrary factor cores -> Not supported in yaml save/load and atom search, only influenceing create_cores!
        self.backCores = backCores

        self.find_atoms()
        self.dimDict = {**{atomColor: 2 for atomColor in self.distributedVariables},
                        **engine.get_dimDict(backCores)}

    def __str__(self):
        outString = "Hybrid Knowledge Base consistent of"
        if self.weightedFormulas:
            outString = outString + "\n######## probabilistic formulas:\n" + "\n".join(
                [ftc.get_formula_headColor(expression[:-1]) + " with weight " + str(expression[-1]) for
                 expression in
                 self.weightedFormulas.values()])
        if self.facts:
            outString = outString + "\n######## logical formulas:\n" + "\n".join(
                [ftc.get_formula_headColor(expression) for expression in self.facts.values()])
        if self.categoricalConstraints:
            outString = outString + "\n######## categorical variables:\n" + "\n".join(
                [key + " selecting one of " + " ".join(self.categoricalConstraints[key]) for key in
                 self.categoricalConstraints]
            )
        if self.backCores:
            outString = outString + "\n######## further cores:\n" + "\n".join(list(self.backCores.keys()))
        return outString

    def find_atoms(self):
        """
        Identifies the atoms of the Knowledge Base
        """
        self.distributedVariables = st.get_all_atom_colors(
            {**{key: self.weightedFormulas[key][:-1] for key in self.weightedFormulas},
             **self.facts})
        for constraintKey in self.categoricalConstraints:
            for atom in self.categoricalConstraints[constraintKey]:
                if atom not in self.distributedVariables:
                    self.distributedVariables.append(atom)
        for eKey in self.evidence:
            if eKey not in self.distributedVariables:
                self.distributedVariables.append(eKey)
        self.distributedVariables = list(self.distributedVariables)

    def from_yaml(self, loadPath):
        modelSpec = stor.load_from_yaml(loadPath)
        if probFormulasKey in modelSpec:
            self.weightedFormulas = modelSpec[probFormulasKey]
        if logFormulasKey in modelSpec:
            self.facts = modelSpec[logFormulasKey]
        if categoricalsKey in modelSpec:
            self.categoricalConstraints = modelSpec[categoricalsKey]
        if evidenceKey in modelSpec:
            self.evidence = modelSpec[evidenceKey]
        self.find_atoms()

    def to_yaml(self, savePath):
        stor.save_as_yaml({
            probFormulasKey: {key: self.weightedFormulas[key][:-1] + [float(self.weightedFormulas[key][-1])] for key in
                              self.weightedFormulas},
            logFormulasKey: self.facts,
            categoricalsKey: self.categoricalConstraints,
            evidenceKey: self.evidence
        }, savePath)

    def include(self, secondHybridKB):
        self.weightedFormulas = {**self.weightedFormulas,
                                 **secondHybridKB.weightedFormulas}
        self.facts = {**self.facts,
                      **secondHybridKB.facts}
        self.categoricalConstraints = {**self.categoricalConstraints,
                                       **secondHybridKB.categoricalConstraints}
        self.evidence = {**self.evidence,
                         **secondHybridKB.evidence}
        self.find_atoms()

    def create_caNetwork(self, hardOnly=False):
        ## Hard Features
        featureDict = {formulaKey: representation.HardPartitionFeature(
            featureColors=[ftc.get_formula_headColor(self.facts[formulaKey])],
            affectedComputationCores=list(
                ftc.create_formula_computation_cores(self.facts[formulaKey]).keys()),
        ) for formulaKey in self.facts}
        computationCoreDict = ftc.create_computation_cores_to_expressionDict(self.facts, coreType=self.coreType)

        ## Soft Features
        if not hardOnly:
            featureDict.update(
                {formulaKey: representation.SingleSoftFeature(
                    featureColor=ftc.get_formula_headColor(self.weightedFormulas[formulaKey][:-1]),
                    affectedComputationCores=list(
                        ftc.create_formula_computation_cores(self.weightedFormulas[formulaKey][:-1]).keys()),
                    # Faster to have a function getting the coreKeys without instantiating the network!
                ) for formulaKey in self.weightedFormulas}
            )
            computationCoreDict.update(
                ftc.create_computation_cores_to_expressionDict(self.weightedFormulas, coreType=self.coreType))

        baseMeasureCoreDict = {
            **ftc.create_formula_evidence_cores(self.evidence, coreType=self.coreType),
            **ctc.create_categorical_cores(self.categoricalConstraints, coreType=self.coreType, addColorSuffixes=True),
            **ctc.create_atomization_cores([atom for atom in self.distributedVariables if "=" in atom],
                                           self.dimDict, coreType=self.coreType),
            **self.backCores
        }

        canParamDict = {
            **{formulaKey: self.weightedFormulas[formulaKey][-1] for formulaKey in self.weightedFormulas},
            **{formulaKey: representation.create_basis_core(name=formulaKey, shape=[2], colors=[
                ftc.get_formula_headColor(self.facts[formulaKey])], numberTuple=(1)
                                                            ) for formulaKey in self.facts}
        }

        return representation.ComputationActivationNetwork(featureDict=featureDict, canParamDict=canParamDict,
                                                           computationCoreDict=computationCoreDict,
                                                           baseMeasureCoreDict=baseMeasureCoreDict,
                                                           distributedVariables=self.distributedVariables,
                                                           coreType=self.coreType)


## For Markov Networks: Find Hard and Soft Features to each core
def create_factor_hardCore(core, outCoreType, outName):
    needed = False
    newCore = engine.get_core(coreType=outCoreType)(colors=core.colors, shape=core.shape, name=outName)
    for posTuple in np.ndindex(*core.shape):
        posDict = {color: posTuple[i] for i, color in enumerate(core.colors)}
        if core[posDict] == 0:
            needed = True
        else:
            newCore[posDict] = 1
    return newCore, needed


def create_factor_softCore(core, outCoreType, outName):
    needed = False
    newCore = engine.get_core(coreType=outCoreType)(colors=core.colors, shape=core.shape, name=outName)
    for posTuple in np.ndindex(*core.shape):
        posDict = {color: posTuple[i] for i, color in enumerate(core.colors)}
        coreValue = core[posDict]
        if coreValue not in [0, 1]:  # Those are handled by hard core
            needed = True
            newCore[posDict] = math.log(coreValue)
    return newCore, needed
