import numpy as np

from tnreason.engine import subscript_creation as subc


def np_random_core(shape, colors, randomEngine, name):
    if randomEngine == "NumpyUniform":
        return NumpyCore(values=np.random.random(size=shape), colors=colors, name=name)
    elif randomEngine == "NumpyNormal":
        return NumpyCore(values=np.random.normal(size=shape), colors=colors, name=name)
    else:
        raise ValueError("Random Engine {} not known for core creation!".format(randomEngine))


class NumpyCore:
    def __init__(self, values=None, colors=None, name=None, shape=None):

        if values is None:  # Empty initialization based on shape
            self.values = np.zeros(shape=shape)
            self.shape = shape
        else:  # Initialization based on values
            self.values = np.array(values)
            self.shape = self.values.shape

        self.colors = colors
        self.name = name

        if len(self.colors) != len(self.values.shape):
            raise ValueError("Number of Colors does not match the Value Shape in Core {}!".format(name))
        if len(self.colors) != len(set(self.colors)):
            raise ValueError("There are duplicate colors in the colors {} of Core {}!".format(colors, name))

        self.index = 0

    def __str__(self):
        return "## Numpy Core " + str(self.name) + " ##\nValues with shape: " + str(self.shape) + "\nColors: " + str(
            self.colors)

    def __getitem__(self, item):
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

    ## For Sampling
    def normalize(self):
        return NumpyCore(1 / np.sum(self.values) * self.values, self.colors, self.name)

    ## For ALS: Reorder Colors and summation
    def reorder_colors(self, newColors):
        self.values = np.einsum(subc.get_reorder_substring(self.colors, newColors), self.values)
        self.colors = newColors

    def sum_with(self, sumCore):
        if set(self.colors) != set(sumCore.colors):
            raise ValueError("Colors of summands {} and {} do not match!".format(self.name, sumCore.name))
        else:
            self.reorder_colors(sumCore.colors)
            return NumpyCore(self.values + sumCore.values, self.colors, self.name)

    def multiply(self, weight, sliceDict=dict()):
        if len(sliceDict) == 0:
            self.values = weight * self.values
            return self
        else:
            subscript = tuple([slice(None) if color not in sliceDict else sliceDict[color] for color in self.colors])
            self.values[tuple(subscript)] = weight * self.values[tuple(subscript)]
            return self

    def exponentiate(self):
        return NumpyCore(np.exp(self.values), self.colors, self.name)

    def build_ln(self):
        return NumpyCore(np.log(self.values), self.colors, self.name)

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

    def calculate_coordinatewise_kl_to(self, secondCore):
        klDivergences = np.empty(self.values.shape)
        for x in np.ndindex(self.values.shape):
            klDivergences[x] = bernoulli_kl_divergence(self.values[x], secondCore.values[x])
        return NumpyCore(values=klDivergences, colors=self.colors, name=str(self.name) + "_kl_" + str(secondCore.name))


class NumpyEinsumContractor:
    def __init__(self, coreDict={}, openColors=[]):
        self.coreDict = {key: coreDict[key].clone() for key in coreDict}
        self.openColors = openColors

    def contract(self):
        substring, coreOrder, colorDict, colorOrder = subc.get_einsum_substring(self.coreDict, self.openColors)
        return NumpyCore(
            np.einsum(substring, *[self.coreDict[key].values for key in coreOrder]),
            [color for color in colorOrder if color in self.openColors])


def bernoulli_kl_divergence(p1, p2):
    """
    Calculates the Kullback Leibler Divergence between two Bernoulli distributions with parameters p1, p2
    """
    if p1 == 0:
        return np.log(1 / (1 - p2))
    elif p1 == 1:
        return np.log(1 / p2)
    return p1 * np.log(p1 / p2) + (1 - p1) * np.log((1 - p1) / (1 - p2))
