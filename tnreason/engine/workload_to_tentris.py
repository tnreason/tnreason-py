from tentris import tentris, Hypertrie

from tnreason.engine import subscript_creation as subc

import numpy as np


def ht_from_rdf(path, tripleColors=["s", "p", "o"], name="KnowledgeGraphCore"):
    tStore = tentris.TripleStore()
    tStore.load_rdf_data(path)
    return HypertrieCore(values=tStore.hypertrie(), colors=tripleColors, name=name)


class HypertrieCore:
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
