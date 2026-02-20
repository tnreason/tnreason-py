import numpy as np

from tnreason.engine import subscript_creation as subc
from tnreason.engine import core_base as cb

def np_random_core(shape, colors, randomEngine, name):
    if randomEngine == "NumpyUniform":
        return NumpyCore(values=np.random.random(size=shape), colors=colors, name=name)
    elif randomEngine == "NumpyNormal":
        return NumpyCore(values=np.random.normal(size=shape), colors=colors, name=name)
    else:
        raise ValueError("Random Engine {} not known for core creation!".format(randomEngine))


class NumpyCore(cb.TensorCore):
    coreType = "NumpyCore"

    def __init__(self, values=None, colors=None, name="NoName", shape=None):
        super().__init__(colors, name, shape)

        if values is None:  # Empty initialization based on shape
            self.values = np.zeros(shape=self.shape).astype(float)
        else:  # Initialization based on values
            self.values = np.array(values)
            self.shape = self.values.shape

        self.index = 0

    def __getitem__(self, item):
        if isinstance(item, dict):
            return self.values[tuple(item.get(color, None) for color in self.colors)]
        elif isinstance(item, slice) and item.start is None:
            """ Then full slize : and Core needs to be just a number """
            assert len(self.colors) == 0
            return float(self.values)
        else:
            return self.values[item]

    def __setitem__(self, sliceDict, value):
        """
        Adds a value onto the slice, not erasing the values before!
        """
        subscript = tuple([slice(None) if color not in sliceDict else sliceDict[color] for color in self.colors])
        onesShape = [self.shape[i] for i, color in enumerate(self.colors) if color not in sliceDict]
        self.values[subscript] = self.values[subscript] + value * np.ones(shape=onesShape)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < np.prod(self.shape):
            indexTuple = np.unravel_index(self.index, self.shape)
            value = self.values[indexTuple]
            self.index += 1
            return (value, {color: indexTuple[i] for i, color in enumerate(self.colors)})
        else:
            self.index = 0
            raise StopIteration

    def clone(self):
        return NumpyCore(self.values.copy(), self.colors.copy(), self.name)  # ! Shallow Copies?

    def contract_with(self, core2): # For usage in Corewise Contractor
        newColors = list(set(self.colors) | set(core2.colors))
        newShapes = [0 for _ in newColors]
        for i, color in enumerate(self.colors):
            newShapes[newColors.index(color)] = self.shape[i]
        for i, color in enumerate(core2.colors):
            newShapes[newColors.index(color)] = core2.shape[i]
        return NumpyEinsumContractor(coreDict={self.name: self, core2.name: core2}, openColors=newColors).contract()

    ## For Sampling
    def normalize(self):
        return NumpyCore(1 / np.sum(self.values) * self.values, self.colors, self.name)

    ## For ALS: Reorder Colors and summation
    def reorder_colors(self, newColors):
        self.values = np.einsum(subc.get_reorder_substring(self.colors, newColors), self.values)
        self.colors = newColors

    def __add__(self, otherCore):
        if set(self.colors) != set(otherCore.colors):
            raise ValueError("Colors of summands {} and {} do not match!".format(self.name, otherCore.name))
        else:
            self.reorder_colors(otherCore.colors)
            return NumpyCore(self.values + otherCore.values, self.colors, self.name)

    def __rmul__(self, scalar):
        self.values = scalar * self.values
        return self

    def slice_multiply(self, weight, sliceDict=dict()):
        subscript = tuple([slice(None) if color not in sliceDict else sliceDict[color] for color in self.colors])
        self.values[tuple(subscript)] = weight * self.values[tuple(subscript)]
        return self

    def get_slice(self, colorEvidenceDict={}):
        newValues = self.values[tuple(colorEvidenceDict.get(color, slice(None))
                                        for i, color in enumerate(self.colors))]
        newColors = [color for color in self.colors if color not in colorEvidenceDict]
        newShape = [self.shape[i] for i, color in enumerate(self.colors) if color not in colorEvidenceDict]
        return NumpyCore(values=newValues, colors=newColors, shape=newShape, name="Sliced_"+self.name)

    def get_argmax(self):
        return {self.colors[i]: maxPos for i, maxPos in
                enumerate(np.unravel_index(np.argmax(self.values.flatten()), self.values.shape))}

    def draw_sample(self, asEnergy=False, temperature=1):
        if asEnergy:
            distribution = np.exp(self.values * 1 / temperature).flatten()
        else:
            distribution = self.values.flatten()
        sample = np.unravel_index(
            np.random.choice(np.arange(np.prod(distribution.shape)), p=distribution / np.sum(distribution)),
            self.values.shape)
        return {color: sample[i] for i, color in enumerate(self.colors)}

class NumpyEinsumContractor:
    def __init__(self, coreDict={}, openColors=[]):
        self.coreDict = {key: coreDict[key].clone() for key in coreDict}
        self.openColors = openColors

    def contract(self):
        substring, coreOrder, colorDict, colorOrder = subc.get_einsum_substring(self.coreDict, self.openColors)
        return NumpyCore(
            np.einsum(substring, *[self.coreDict[key].values for key in coreOrder]),
            [color for color in colorOrder if color in self.openColors])