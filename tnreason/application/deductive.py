from tnreason import engine
from tnreason import reasoning
from tnreason import representation

from tnreason.application import script_transform as st
from tnreason.application import formulas_to_cores as ftc

entailedString = "entailed"
contradictingString = "contradicting"
contingentString = "contingent"


class InferenceProvider(engine.EngineUser):
    """
    Answering queries on a distribution by contracting its cores.
    """

    def __init__(self, distribution, **engineSpec):
        """
        * distribution: Needs to support create_cores(), get_partition_function()
        """
        super().__init__(**engineSpec)
        self.distribution = distribution

    def ask_features(self, featureDict, computationCoreDict=dict(),
                     forwardInferenceSpec={"inferenceMethod": "ForwardContractor"}):
        if isinstance(self.distribution, representation.ComputationActivationNetwork):
            caNetwork = self.distribution.clone()
        else:
            caNetwork = self.distribution.create_caNetwork()
        caNetwork.include_features(featureDict=featureDict,
                                   computationCores=computationCoreDict)

        fInferer = reasoning.get_inferer(forwardInferenceSpec["inferenceMethod"])(caNetwork=caNetwork)
        fInferer.infer_meanParams(featureDict.keys())

        return {featureKey: fInferer.meanParamDict[featureKey] for featureKey in featureDict}

    def ask_constraint(self, constraint):
        probability = self.ask(constraint, evidenceDict={})
        if probability > 0.9999:
            return entailedString
        elif probability == 0:
            return contradictingString
        else:
            return contingentString

    def ask(self, queryFormula, evidenceDict={}):
        formulaComCores = ftc.create_formula_computation_cores(queryFormula)
        featureDict = {
            "queryFeat": representation.HardPartitionFeature(featureColors=[ftc.get_formula_headColor(queryFormula)],
                                                             affectedComputationCores=list(formulaComCores.keys()))}
        contracted2 = self.ask_features(featureDict, formulaComCores)["queryFeat"]
        queryColor = ftc.get_formula_headColor(queryFormula)

        contracted = engine.contract(
            coreDict={
                **self.distribution.create_cores(),
                **ftc.create_formula_evidence_cores(evidenceDict),
                **ftc.create_formula_computation_cores(queryFormula)
            },
            contractionMethod=self.contractionMethod, openColors=[queryColor])
        return contracted[{queryColor: 1}] / (contracted[{queryColor: 0}] + contracted[{queryColor: 1}])

    def query(self, variableList, evidenceDict={}):
        """
        While colorList is a list of colors, evidenceDict entries will get atom suffix!
        """
        return engine.normalize(coreDict={**self.distribution.create_cores(),
                                          **ftc.create_formula_evidence_cores(evidenceDict)},
                                inColors=[], outColors=st.add_color_suffixes(variableList),
                                contractionMethod=self.contractionMethod
                                )

    def exact_map_query(self, variableList, evidenceDict={}):
        """
        When distributionCore is a
            * PolynomialCore, uses gurobi optimizer on integer linear program
            * NumpyCore uses the argmax method of numpy
        """
        return st.drop_color_suffixes_from_assignment(self.query(variableList, evidenceDict).get_argmax())

    def search_mode(self, variableList=None, optimizationMethod="numpyArgmax", **specDict):
        """
        SpecDict: Includes engineSpec, when empty takes the
        """
        if not "coreType" in specDict:
            specDict["coreType"] = self.coreType
        if not "contractionMethod" in specDict:
            specDict["contraction"] = self.contractionMethod

        variableList = variableList or self.distribution.distributedVariables
        if optimizationMethod in reasoning.coreOptimizationMethods:
            return reasoning.core_based_optimize(self.distribution.create_cores(),
                                                 variableList=variableList,
                                                 optimizationMethod=optimizationMethod,
                                                 **specDict)
        elif optimizationMethod in reasoning.energyOptimizationMethods:
            return reasoning.energy_based_optimize(energyDict=self.distribution.get_energy_dict(),
                                                   variableList=variableList,
                                                   optimizationMethod=optimizationMethod,
                                                   **specDict)

    def draw_samples(self, sampleNum, nameList=None, method="forwardSampling", dfOutput=False):
        """
        Initializes a Sampler being an iteratable core
        """
        if nameList is None:
            colorList = self.distribution.distributedVariables
        else:
            colorList = st.add_color_suffixes(nameList)
        if method in reasoning.energySamplingMethods:
            sampler = reasoning.get_energy_based_sampler(self.distribution.get_energy_dict(), samplingMethod=method,
                                                         sampleNum=sampleNum,
                                                         colors=colorList,
                                                         contractionMethod=self.contractionMethod,
                                                         coreType=self.coreType)
        elif method in reasoning.coreSamplingMethods:
            sampler = reasoning.get_core_based_sampler(self.distribution.create_cores(), samplingMethod=method,
                                                       sampleNum=sampleNum,
                                                       colors=colorList,
                                                       contractionMethod=self.contractionMethod,
                                                       coreType=self.coreType)
        else:
            raise ValueError("Sampling Method {} not implemented.".format(method))
        if dfOutput:
            return engine.convert(sampler, "PandasCore").values.astype("int64")
        else:
            # Samples still in colors! In draw_sample reduced to names
            return sampler

    def draw_sample(self, names=None, method="forwardSampling"):
        sampler = self.draw_samples(sampleNum=1, nameList=names, method=method, dfOutput=False)
        iter(sampler)
        return st.drop_color_suffixes_from_assignment(next(sampler)[1])
