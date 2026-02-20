from tentris import tentris, Hypertrie

from tnreason.engine import subscript_creation as subc

from tnreason.representation import suffixes as suf

import numpy as np


class TentrisTripleStoreTermCore:
    """
    Term Cores are iterateable, but not contractable cores.
    They differ from other cores by their interpretationDict, which stores string identifiers to each term variable and index.
    """

    def __init__(self, startInterpretationDict=dict(), spoVariables=["s", "p", "o"], name="TripleStoreCore"):
        self.tStore = tentris.TripleStore()
        self.interpretationDict = startInterpretationDict
        self.projectedVariables = spoVariables
        for variable in self.projectedVariables:
            if variable not in self.interpretationDict:
                self.interpretationDict[variable] = []

        ## Mimicking Properties of TensorCores for iterators
        self.colors = [variable + suf.terVarSuf for variable in self.projectedVariables]
        self.shape = [len(self.interpretationDict[variable]) for variable in self.projectedVariables]
        self.name = name

    def load_rdf_data(self, dataPath):
        self.tStore.load_rdf_data(dataPath)

    def adjust_interpretationsDict(self):
        for pos, value in self.tStore.hypertrie():
            for i, variable in enumerate(self.projectedVariables):
                identifier = str(self.tStore.try_get_resource(pos[i]))
                if identifier not in self.interpretationDict[variable]:
                    self.interpretationDict[variable].append(identifier)
        self.shape = [len(self.interpretationDict[variable]) for variable in self.projectedVariables]

    def __iter__(self):
        self.iterator = iter(self.tStore.hypertrie())
        return self

    def __next__(self):
        pos, value = next(self.iterator)
        return (value, {variable + suf.terVarSuf: self.interpretationDict[variable].index(
            str(self.tStore.try_get_resource(pos[i])))
            for i, variable in enumerate(self.projectedVariables)})

    def eval_query(self, queryString, variables):

        ## First evaluation to get the shape of the result
        querySolution = self.tStore.eval_sparql_query(queryString)

        queryInterpretationDict = dict()
        for variable in variables:
            if variable in self.interpretationDict:
                queryInterpretationDict[variable] = self.interpretationDict[variable]
            else:
                queryInterpretationDict[variable] = []

        for pos, value in iter(querySolution):
            for i, variable in enumerate(variables):
                identifierString = str(pos[querySolution.projected_variables[i]])
                if identifierString not in queryInterpretationDict[variable]:
                    queryInterpretationDict[variable].append(identifierString)

        shape = [len(queryInterpretationDict[variable]) for variable in variables]

        ## Second evaluation into iterator
        return TentrisSPARQLEvaluationTermCore(self.tStore.eval_sparql_query(queryString), queryInterpretationDict,
                                               variables=variables, shape=shape)


class TentrisSPARQLEvaluationTermCore:

    def __init__(self, querySolution, startInterpretationDict, variables=[], shape=[], name=None):
        self.querySolution = querySolution
        self.interpretationDict = dict()
        self.projectedVariables = querySolution.projected_variables
        self.variables = variables
        for variable in variables:
            if variable in startInterpretationDict:
                self.interpretationDict[variable] = startInterpretationDict[variable]
            else:
                self.interpretationDict[variable] = []

        self.colors = [variable + suf.terVarSuf for variable in variables]
        self.shape = shape
        self.name = name

    def __iter__(self):
        self.iterator = iter(self.querySolution)
        return self

    def __next__(self):
        pos, value = next(self.iterator)
        return (value,
                {variable + suf.terVarSuf: self.interpretationDict[variable].index(str(
                    pos[self.projectedVariables[i]])) for i, variable in
                    enumerate(self.variables)})


class HypertrieCore:
    coreType="HypertrieCore"
    def __init__(self, values=None, colors=None, name=None, shape=None, dtype=float):
        self.colors = colors
        self.name = name

        if values is None:
            self.values = Hypertrie(dtype=dtype, depth=len(shape))
            self.shape = shape
        elif isinstance(values, Hypertrie):
            self.values = values
            self.get_shape()

        self.index = 0

    def __str__(self):
        return "## Hypertrie Core " + str(self.name) + "\nColors: " + str(self.colors)

    def __getitem__(self, item):
        if isinstance(item, dict):
            return self.values[tuple(item.get(color, None) for color in self.colors)]
        else:
            return self.values[item]

    def __setitem__(self, sliceDict, value):
        subscript = tuple([slice(None) if color not in sliceDict else sliceDict[color] for color in self.colors])
        self.values[subscript] = self.values[subscript] + value

    def __iter__(self):
        self.iterator = iter(self.values)
        return self

    def __next__(self):
        pos, value = next(self.iterator)
        return (value, {color: pos[i] for i, color in enumerate(self.colors)})

    def get_shape(self):
        shape = np.zeros(self.values.depth)
        for entry in self.values:
            for i in range(len(shape)):
                if entry[0][i] + 1 > shape[i]:
                    shape[i] = entry[0][i] + 1
        self.shape = [int(dim) for dim in shape]


class HypertrieContractor:
    def __init__(self, coreDict, openColors):
        for key in coreDict:
            if not isinstance(coreDict[key], HypertrieCore):
                raise ValueError(
                    "Hypertrie Contractions works only for Hypertrie, but got core {} of type {}!".format(key, type(
                        coreDict[key])))
        self.coreDict = coreDict
        self.openColors = openColors

    def einsum(self):
        substring, coreOrder, colorDict, colorOrder = subc.get_einsum_substring(self.coreDict, self.openColors)
        with tentris.einsumsum(subscript=substring, operands=[self.coreDict[key].values for key in coreOrder]) as e:
            resultValues = Hypertrie(dtype=e.result_dtype, depth=e.result_depth)
            e.try_extend_hypertrie(resultValues)

        return HypertrieCore(values=resultValues,
                             colors=[color for color in colorOrder if color in self.openColors])
