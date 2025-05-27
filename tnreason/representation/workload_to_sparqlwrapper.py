from SPARQLWrapper import SPARQLWrapper, JSON

from tnreason.representation import suffixes as suf

from tnreason import engine


def queries_to_cores(endpointString, queryDict, startInterpretationDict=dict(), coreType=None):
    """
    Evaluates a dictionary of queries with a single runCore, which interpretationDict is maintained throught all queries
    """
    runCore = SPARQLWrapperTermCore(endpointString=endpointString, startInterpretationDict=startInterpretationDict)
    queryCoresDict = dict()
    for nameKey in queryDict:
        runCore.run_query(queryString=queryDict[nameKey], name=nameKey,
                          adjustInterpretation=True)
        queryCoresDict[nameKey] = engine.convert(runCore, outCoreType=coreType)
    return queryCoresDict, runCore.interpretationDict


def query_to_core(endpointString, queryString, startInterpretationDict=dict(),
                  name="ExampleQuery", coreType=None):
    queryCoresDict, interpretationDict = queries_to_cores(endpointString, {name: queryString},
                                                          startInterpretationDict=startInterpretationDict,
                                                          coreType=coreType)
    return queryCoresDict[name], interpretationDict


class SPARQLWrapperTermCore:
    """
    Term Cores are iterateable, but not contractable cores.
    They differ from other cores by their interpretationDict, which stores string identifiers to each term variable and index.
    """

    def __init__(self, endpointString, startInterpretationDict=dict(), adjustWhileIterate=True):
        self.sparql = SPARQLWrapper(endpointString)
        self.interpretationDict = startInterpretationDict
        self.adjustWhileIterate = adjustWhileIterate

    def run_query(self, queryString, name="Query", adjustInterpretation=True):
        self.sparql.setQuery(queryString)
        self.sparql.setReturnFormat(JSON)

        self.querySolution = self.sparql.query().convert()
        self.projectedVariables = [str(variable) for variable in self.querySolution["head"]["vars"]]

        self.interpretationDict.update(
            {variable: [] for variable in self.projectedVariables if variable not in self.interpretationDict})

        if adjustInterpretation:
            self.adjust_interpretationsDict()

        ## Mimicking Properties of TensorCores for iterators
        self.colors = [variable + suf.terVarSuf for variable in self.projectedVariables]
        self.shape = [len(self.interpretationDict[variable]) for variable in self.projectedVariables]
        self.name = name

    def adjust_interpretationsDict(self):
        """
        Necessary before shape is used
        """
        for entry in iter(self.querySolution["results"]["bindings"]):
            for variable in self.projectedVariables:
                identifierString = str(entry[variable]["value"])
                if identifierString not in self.interpretationDict[variable]:
                    self.interpretationDict[variable].append(identifierString)
        self.shape = [len(self.interpretationDict[variable]) for variable in self.projectedVariables]
        self.adjustWhileIterate = False

    def __iter__(self):
        self.solutionIterator = iter(self.querySolution["results"]["bindings"])
        return self

    def __next__(self):
        entry = next(self.solutionIterator)

        if self.adjustWhileIterate:
            for variable in self.projectedVariables:
                identifierString = str(entry[variable]["value"])
                if identifierString not in self.interpretationDict[variable]:
                    self.interpretationDict[variable].append(identifierString)

        return (1, {
            variable + suf.terVarSuf: self.interpretationDict[variable].index(str(entry[variable]["value"]))
            for variable in self.projectedVariables})
