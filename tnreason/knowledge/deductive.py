from tnreason import engine
from tnreason import encoding
from tnreason import algorithms

entailedString = "entailed"
contradictingString = "contradicting"
contingentString = "contingent"


class InferenceProvider(engine.EngineUser):
    """
    Answering queries on a distribution by contracting its cores.
    """

    def __init__(self, distribution, **engineSpec):
        """
        * distribution: Needs to support create_cores() and get_partition_function()
        """
        super().__init__(**engineSpec)
        self.distribution = distribution

    def ask_constraint(self, constraint):
        probability = self.ask(constraint, evidenceDict={})
        if probability > 0.9999:
            return entailedString
        elif probability == 0:
            return contradictingString
        else:
            return contingentString

    def ask(self, queryFormula, evidenceDict={}):
        queryColor = encoding.get_formula_color(queryFormula)

        contracted = engine.contract(
            coreDict={
                **self.distribution.create_cores(),
                **encoding.create_atom_evidence_cores(evidenceDict),
                **encoding.create_raw_formula_cores(queryFormula)
            },
            contractionMethod=self.contractionMethod, openColors=[queryColor])
        return contracted[{queryColor: 1}] / (contracted[{queryColor: 0}] + contracted[{queryColor: 1}])

    def query(self, variableList, evidenceDict={}):
        return engine.normate(coreDict={**self.distribution.create_cores(),
                                        **encoding.create_atom_evidence_cores(evidenceDict)},
                              inColors=[], outColors=variableList,
                              contractionMethod=self.contractionMethod
                              )

    def exact_map_query(self, variableList, evidenceDict={}):
        """
        When distributionCore is a
            * PolynomialCore, uses gurobi optimizer on integer linear program
            * NumpyCore uses the argmax method of numpy
        """
        return self.query(variableList, evidenceDict).get_argmax()

    def search_mode(self, variableList=None, optimizationMethod="numpyArgmax", **specDict):
        """
        SpecDict: Includes engineSpec, when empty takes the
        """
        if not "coreType" in specDict:
            specDict["coreType"] = self.coreType
        if not "contractionMethod" in specDict:
            specDict["contraction"] = self.contractionMethod

        variableList = variableList or self.distribution.distributedVariables
        if optimizationMethod in algorithms.coreOptimizationMethods:
            return algorithms.core_based_optimize(self.distribution.create_cores(),
                                                  variableList=variableList,
                                                  optimizationMethod=optimizationMethod,
                                                  **specDict)
        elif optimizationMethod in algorithms.energyOptimizationMethods:
            return algorithms.energy_based_optimize(energyDict=self.distribution.get_energy_dict(),
                                                    variableList=variableList,
                                                    optimizationMethod=optimizationMethod,
                                                    **specDict)

    def draw_samples(self, sampleNum, colors=None, method="forwardSampling", dfOutput=False):
        """
        Initializes a Sampler being an iteratable core
        """
        if colors is None:
            colors = self.distribution.distributedVariables
        if method in algorithms.energySamplingMethods:
            sampler = algorithms.get_energy_based_sampler(self.distribution.get_energy_dict(), samplingMethod=method,
                                                          sampleNum=sampleNum,
                                                          colors=colors,
                                                          contractionMethod=self.contractionMethod,
                                                          coreType=self.coreType)
        elif method in algorithms.coreSamplingMethods:
            sampler = algorithms.get_core_based_sampler(self.distribution.create_cores(), samplingMethod=method,
                                                        sampleNum=sampleNum,
                                                        colors=colors,
                                                        contractionMethod=self.contractionMethod,
                                                        coreType=self.coreType)
        else:
            raise ValueError("Sampling Method {} not implemented.".format(method))
        if dfOutput:
            return engine.convert(sampler, "PandasCore").values.astype("int64")
        else:
            return sampler

    def draw_sample(self, colors=None, method="forwardSampling"):
        sampler = self.draw_samples(sampleNum=1, colors=colors, method=method, dfOutput=False)
        iter(sampler)
        return next(sampler)[1]
