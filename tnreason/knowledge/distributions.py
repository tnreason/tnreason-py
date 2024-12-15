from tnreason import encoding
from tnreason import engine
import math

probFormulasKey = "weightedFormulas"
logFormulasKey = "facts"
categoricalsKey = "categoricalConstraints"
evidenceKey = "evidence"


class DistributionBase:
    """
    Distributions have two methods:
        * create_cores(): returning the factor cores -> customized!
        * get_partition_function(allAtoms): returning the partition function given the atomic variables of interest
    """

    def __init__(self, **specDict):
        self.partitionFunction = specDict.get("partitionFunction", None)
        self.coreType = specDict.get("coreType", None)
        self.contractionMethod = specDict.get("contractionMethod", None)

    def get_partition_function(self, addDimDict={}):
        if self.partitionFunction is None:
            self.partitionFunction = engine.contract(self.create_cores(), openColors=[], contractionMethod=self.contractionMethod)[
                                     :]
        return math.prod(
            [addDimDict[color] for color in addDimDict if color not in self.dimDict]) * self.partitionFunction


class MarkovNetwork(DistributionBase):
    """
    Interprets a Tensor Network as a distribution
    """

    def __init__(self, coreDict, **specDict):
        super().__init__(**specDict)

        self.coreDict = coreDict
        self.dimDict = engine.get_dimDict(coreDict)

    def create_cores(self):
        return self.coreDict


def get_empirical_distribution(sampleDf, atomColors=None, interpretation="atomic", dimensionsDict=None):
    """
    Returns an empirical distributions as a MarkovNetwork
        * sampleDf: pd.DataFrame containing the samples defining the empirical distributions
        * atomKeys: List of columns of sampleDf to be recognized as atoms
        * interpretation: Specifies the interpretation of the entries of sampleDf
            - "atomic": Variables have dimension 2 and entries in [0,1] are the probability of the atom holding.
            - "categorical": Variables have dimension m specified in dimensionsDict and entries are the certain value of the variable in [m]
    """
    if atomColors is not None:
        sampleDf = sampleDf[atomColors]
    else:
        atomColors = list(sampleDf.columns)
    if "value" not in sampleDf.columns:
        sampleDf["value"] = 1
    return MarkovNetwork(encoding.create_data_cores(sampleDf, atomKeys=atomColors,
                                                    interpretation=interpretation,
                                                    dimensionsDict=dimensionsDict),
                         partitionFunction=sampleDf["value"].sum())


class HybridKnowledgeBase(DistributionBase):
    """
    Inferable (by HybridInferer) Knowledge Base. Generalizes Markov Logic Network by further dedicated cores
    * dimensionDict: Dictionary of dimensions for in the formulas appearing categorical variables
    """

    def __init__(self, weightedFormulas={}, facts={}, categoricalConstraints={}, evidence={}, backCores={}, **specDict):
        super().__init__(**specDict)

        self.weightedFormulas = {key: weightedFormulas[key][:-1] + [float(weightedFormulas[key][-1])] for key in
                                 weightedFormulas}
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
                [encoding.get_formula_color(expression[:-1]) + " with weight " + str(expression[-1]) for expression in
                 self.weightedFormulas.values()])
        if self.facts:
            outString = outString + "\n######## logical formulas:\n" + "\n".join(
                [encoding.get_formula_color(expression) for expression in self.facts.values()])
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
        self.distributedVariables = encoding.get_all_atoms(
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
        modelSpec = encoding.load_from_yaml(loadPath)
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
        encoding.storage.save_as_yaml({
            probFormulasKey: self.weightedFormulas,
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

    def create_cores(self):
        return {**encoding.create_formulas_cores({**self.weightedFormulas, **self.facts}, coreType=self.coreType),
                **encoding.create_atom_evidence_cores(self.evidence, coreType=self.coreType),
                **encoding.create_categorical_cores(self.categoricalConstraints, coreType=self.coreType),
                **encoding.create_atomization_cores([atom for atom in self.distributedVariables if "=" in atom],
                                                    self.dimDict, coreType=self.coreType),
                **self.backCores}

    ### Special to Hybrid Knowledge Bases
    def create_hard_cores(self):
        """
        Returns the cores posing hard logical constraints on the worlds to be models
        """
        return {**encoding.create_formulas_cores(self.facts),
                **encoding.create_atom_evidence_cores(self.evidence),
                **encoding.create_categorical_cores(self.categoricalConstraints),
                **self.backCores}

    def is_satisfiable(self):
        """
        Decides whether the Knowledge Base is satisfiable, i.e. whether a model exists
        """
        return engine.contract(coreDict=self.create_hard_cores(),
                               openColors=[]).values > 0

    ## Energy Representation
    def get_energy_dict(self, cutoffWeight=100):
        weightedEnergyDict = {
            formulaKey: [{**encoding.create_raw_formula_cores(self.weightedFormulas[formulaKey][:-1]),
                          **encoding.create_formula_head(self.weightedFormulas[formulaKey][:-1],
                                                         headType="truthEvaluation")
                          }, self.weightedFormulas[formulaKey][-1]] for formulaKey in self.weightedFormulas}
        factsEnergyDict = {formulaKey: [{**encoding.create_raw_formula_cores(self.facts[formulaKey]),
                                         **encoding.create_formula_head(self.facts[formulaKey],
                                                                        headType="truthEvaluation")
                                         }, cutoffWeight] for formulaKey in
                           self.facts}
        constraintsEnergyDict = {constraintKey: [
            encoding.create_constraintCoresDict(self.categoricalConstraints[constraintKey], constraintKey),
            cutoffWeight] for constraintKey in self.categoricalConstraints}

        return {**weightedEnergyDict, **factsEnergyDict, **constraintsEnergyDict}
