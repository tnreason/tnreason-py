from tnreason import engine
from tnreason import encoding
from tnreason import algorithms

entailedString = "entailed"
contradictingString = "contradicting"
contingentString = "contingent"


class InferenceProvider:
    """
    Answering queries on a distribution by contracting its cores.
    """

    def __init__(self, distribution, contractionMethod=engine.defaultContractionMethod):
        """
        * distribution: Needs to support create_cores() and get_partition_function()
        """
        self.distribution = distribution
        self.contractionMethod = contractionMethod

    def ask_constraint(self, constraint):
        probability = self.ask(constraint, evidenceDict={})
        if probability > 0.9999:
            return entailedString
        elif probability == 0:
            return contradictingString
        else:
            return contingentString

    def ask(self, queryFormula, evidenceDict={}):

        contracted = engine.contract(
            coreDict={
                **self.distribution.create_cores(),
                **encoding.create_atom_evidence_cores(evidenceDict),
                **encoding.create_raw_formula_cores(queryFormula)
            },
            method=self.contractionMethod, openColors=[encoding.get_formula_color(queryFormula)]).values

        return contracted[1] / (contracted[0] + contracted[1])

    def query(self, variableList, evidenceDict={}):
        return engine.contract(method=self.contractionMethod, coreDict={
            **self.distribution.create_cores(),
            **encoding.create_atom_evidence_cores(evidenceDict),
        }, openColors=variableList).normalize()

    def exact_map_query(self, variableList, evidenceDict={}):
        """
        When distributionCore is a
            * PolynomialCore, uses gurobi optimizer on integer linear program
            * NumpyCore uses the argmax method of numpy
        """
        distributionCore = self.query(variableList, evidenceDict)
        return distributionCore.get_argmax()

    def forward_sample(self, variableList, dimDict=None):
        """
        To Do: Include in draw_samples as special case
        """
        sampler = algorithms.ForwardSampleCore(self.distribution, dimDict=dimDict, sampleColors=variableList)
        return next(sampler)[1]

    def draw_samples(self, sampleNum, variableList=None, outType="int64", method="ForwardSampling"):
        """
        Initializes a Sampler being an iteratable core
        To Do: Provide options like Energy-based and flexibilize output core to be any (most efficient: Sparse Core).
        """
        if variableList is None:
            variableList = self.distribution.distributedVariables
        if method == "ForwardSampling":
            sampler = algorithms.ForwardSampleCore(self.distribution, sampleColors=variableList, sampleNum=sampleNum)
        else:
            raise ValueError("Sampling Method {} not implemented.".format(method))
        return engine.convert(sampler, "PandasCore").values[variableList].astype(outType)

    ## Energy-based: Creates always a full sample!
    # Can be used for sample (temperatures converging to 1) or for maximum search (temperature converging to 0)
    ## TO DO: Add support for evidenceDict
    def energy_based_sample(self, method, temperatureList=[1 for i in range(10)]):
        return algorithms.optimize_energy(energyDict={**self.distribution.get_energy_dict()},
                                          colors=list(self.distribution.get_dimension_dict().keys()),
                                          dimDict=self.distribution.get_dimension_dict(),
                                          method=method,
                                          temperatureList=temperatureList)
