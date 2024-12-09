import numpy as np


class PolynomialCore:
    """
    :values: Storing the polynomial by a list of tuples, each representing a weighted monomial by
        - value: Weight of the monomial
        - positionDict: Dictionary of variables in the polynomial,
            - each key is the name of a categorical variable X
            - its value k specifies the variable to X==k
    Each monomial seen as a tensor is specified by a weighted trivial slice.
    """

    def __init__(self, values=None, colors=None, name=None, shape=None):

        if values is None:  # Empty intialization
            self.values = []
        else:  # Initialization based on values
            self.values = values

        self.colors = colors
        self.name = name

        self.shape = shape

    def __str__(self):
        return "## Polynomial Core " + str(self.name) + " ##\nValues: " + str(self.values) + "\nColors: " + str(
            self.colors)

    def __getitem__(self, item):
        if isinstance(item, int):
            item = [item]
        value = 0
        for entry in self.values:
            if agreeing_dicts(entry[1], {color: item[i] for i, color in enumerate(self.colors)}):
                value += entry[0]
        return value

    def __setitem__(self, sliceDict, value):
        self.values.append((value, sliceDict))

    def __iter__(self):
        return iter(self.values)

    def clone(self):
        return PolynomialCore(self.values.copy(), self.colors.copy(), self.name, shape=self.shape)  # ! Shallow Copies?

    def contract_with(self, core2):
        newColors = list(set(self.colors) | set(core2.colors))
        newShapes = [0 for color in newColors]
        for i, color in enumerate(self.colors):
            newShapes[newColors.index(color)] = self.shape[i]
        for i, color in enumerate(core2.colors):
            newShapes[newColors.index(color)] = core2.shape[i]

        return PolynomialCore(
            values=slice_contraction(self.values, core2.values),
            shape=newShapes,
            colors=newColors,
            name=str(self.name) + "_" + str(core2.name)
        )

    def reduce_colors(self, newColors):
        newValues = []
        for j in range(len(self.values)):
            newValues.append((np.prod([self.shape[k] for k, col in enumerate(self.colors) if
                                       col not in self.values[j][1] and col not in newColors]) * self.values[j][0],
                              {key: self.values[j][1][key] for key in self.values[j][1] if key in newColors}))
        self.values = newValues
        self.shape = [self.shape[k] for k, col in enumerate(self.colors) if col in newColors]
        self.colors = newColors

    def add_identical_slices(self):
        newSlices = []
        alreadyFound = []
        while len(self.values) != 0:
            val, pos = self.values.pop()
            if pos not in alreadyFound:
                alreadyFound.append(pos)
                for (val2, pos2) in self.values:
                    if pos == pos2:
                        val += val2
                newSlices.append((val, pos))
        self.values = newSlices

    def multiply(self, weight, sliceDict=dict()):

        """
        Cannot handle yet situation of nans in sliceDict
        """
        if len(sliceDict) == 0:
            self.values = [(weight * val, posDict) for val, posDict in self]
            return self
        else:
            self.values = [(weight * val, posDict) if agreeing_dicts(sliceDict, posDict) else (val, posDict) for
                           val, posDict in self]
            return self

    def sum_with(self, sumCore):
        colorsShapeDict = {**{color: self.shape[i] for i, color in enumerate(self.colors)},
                           **{color: sumCore.shape[i] for i, color in enumerate(sumCore.colors)}}
        return PolynomialCore(values=self.values + sumCore.values,
                              shape=list(colorsShapeDict.values()), colors=list(colorsShapeDict.keys()),
                              name=self.name)

    def enumerate_slices(self, enumerationColor="j"):
        self.colors = self.colors + [enumerationColor]
        self.values = [(entry[0], {**entry[1], enumerationColor: i}) for i, entry in enumerate(self.values)]
        self.shape = self.shape + [len(self.values)]

    def reorder_colors(self, newColors):
        if set(self.colors) == set(newColors):
            self.colors = newColors
        else:
            raise ValueError("Reordering of Colors in Core {} not possible, since different!".format(self.name))

    def get_argmax(self, method="gurobi"):
        """
        Works for all iterateable cores, but only for PolynomialCore efficiently, since supporting slice sparsity
        """
        if method == "gurobi":
            from tnreason.engine import workload_to_gurobi as ptg
            return ptg.optimize_gurobi_model(ptg.core_to_gurobi_model(binarize_polyCore(self)))
        else:
            raise ValueError("Maximation Method {} not implemented for PolynomialCore!".format(method))


def slice_contraction(slices1, slices2):
    slices = []
    for (val1, pos1) in slices1:
        for (val2, pos2) in slices2:
            if agreeing_dicts(pos1, pos2):
                slices.append((val1 * val2, {**pos1, **pos2}))
    return slices


def agreeing_dicts(pos1, pos2):
    for key in pos1:
        if key in pos2:
            if pos1[key] != pos2[key]:
                return False
    return True


def binarize_polyCore(polyCore):
    ## Need to further enforce that states with concurring atoms cannot be selected!

    binarizedColors = []
    for i, color in enumerate(polyCore.colors):
        if polyCore.shape[i] > 2:
            binarizedColors = binarizedColors + [color + "_" + str(j) for j in range(polyCore.shape[i])]
        else:
            binarizedColors.append(color)

    binarizedValues = []
    for weight, posDict in polyCore.values:
        binPosDict = {}
        for color in posDict:
            if polyCore.shape[polyCore.colors.index(color)] > 2:
                binPosDict.update(
                    {**{color + "_" + str(i): 0 for i in range(polyCore.shape[polyCore.colors.index(color)]) if
                        i != posDict[color]},
                     color + "_" + str(posDict[color]): 1})
            else:
                binPosDict[color] = posDict[color]
        binarizedValues.append((weight, binPosDict))

    return PolynomialCore(values=binarizedValues,
                          shape=[2 for i in range(len(binarizedColors))],
                          colors=binarizedColors)
