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
                              method=self.contractionMethod
                              )

    def exact_map_query(self, variableList, evidenceDict={}):
        """
        When distributionCore is a
            * PolynomialCore, uses gurobi optimizer on integer linear program
            * NumpyCore uses the argmax method of numpy
        """
        return self.query(variableList, evidenceDict).get_argmax()

    def draw_samples(self, sampleNum, variableList=None, outType="int64", method="ForwardSampling"):
        """
        Initializes a Sampler being an iteratable core
        To Do: Provide options like Energy-based and flexibilize output core to be any (most efficient: Sparse Core).
        """
        if variableList is None:
            variableList = self.distribution.distributedVariables
        if method == "ForwardSampling":
            sampler = algorithms.ForwardSampleCore(self.distribution, sampleNum=sampleNum, colors=variableList,
                                                   contractionMethod=self.contractionMethod,
                                                   coreType=self.coreType)
        else:
            raise ValueError("Sampling Method {} not implemented.".format(method))
        return engine.convert(sampler, "PandasCore").values[variableList].astype(
            outType)  # Flexibilize to arbitrary output CoreType!

    def forward_sample(self, variableList, dimDict=None):
        """
        To Do: Include in draw_samples as special case
        """
        sampler = algorithms.ForwardSampleCore(self.distribution, dimDict=dimDict, colors=variableList)
        iter(sampler)
        return next(sampler)[1]

    ## Energy-based: Creates always a full sample!
    # Can be used for sample (temperatures converging to 1) or for maximum search (temperature converging to 0)
    ## TO DO: Add support for evidenceDict
    def energy_based_sample(self, method, temperatureList=[1 for i in range(10)]):
        return algorithms.optimize_energy(energyDict={**self.distribution.get_energy_dict()},
                                          colors=list(self.distribution.distributedVariables),
                                          dimDict={color: 2 for color in self.distribution.distributedVariables},
                                          method=method,
                                          temperatureList=temperatureList)
