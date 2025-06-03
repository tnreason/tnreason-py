import pandas as pd

from tnreason.engine import core_base as cb

from tnreason.representation import suffixes as suf

import numpy as np

defaultValueColumnString = "values"
defaultNanValue = -1


class PandasCore(cb.TensorCore):
    coreType = "PandasCore"

    def __init__(self, values=None, colors=None, name="NoName", shape=None, valueColumn=defaultValueColumnString,
                 nanValue=defaultNanValue):

        super().__init__(colors, name, shape)

        if values is None:  # Empty initialization based on colors
            self.values = pd.DataFrame(columns=colors)
        else:  # Initialization based on values
            self.values = pd.DataFrame(values)

        self.valueColumn = valueColumn
        if not valueColumn in self.values.columns:
            self.values[valueColumn] = 1

        self.nanValue = nanValue
        self.values = self.values.fillna(nanValue)

        self.index = 0

    def __getitem__(self, item):
        if isinstance(item, dict):
            value = 0
            for j, row in self.values.iterrows():
                if all([row[col] == item[col] or row[col] == self.nanValue for col in self.colors]):
                    value = value + row[self.valueColumn]
            return value
        if isinstance(item, int):
            item = [item]
        checkDict = {color: item[i] for i, color in enumerate(self.colors)}
        value = 0
        for j, row in self.values.iterrows():
            if all([row[col] == checkDict[col] or row[col] == self.nanValue for col in self.colors]):
                value = value + row[self.valueColumn]
        return value

    def __setitem__(self, sliceDict, value):
        new_row = pd.DataFrame(
            {**{color: [sliceDict[color]] for color in sliceDict},
             **{color: [self.nanValue] for color in self.colors if color not in sliceDict},
             self.valueColumn: [value]})
        self.values = pd.concat([self.values, new_row], ignore_index=True)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.values):
            rowDict = self.values.iloc[self.index].to_dict()
            scalar = rowDict.pop(self.valueColumn)
            self.index += 1
            return (scalar, {color: int(rowDict[color]) for color in rowDict if rowDict[color] != self.nanValue})
        else:
            self.index = 0
            raise StopIteration

    def clone(self):
        return PandasCore(self.values.copy(deep=True), self.colors, self.name, self.shape)

    def contract_with(self, core2):
        core2.values = core2.values.rename(columns={core2.valueColumn: self.valueColumn + "_sec"})
        colorsShapeDict = {**{color: self.shape[i] for i, color in enumerate(self.colors)},
                           **{color: core2.shape[i] for i, color in enumerate(core2.colors)}}
        preValues = self.values.merge(core2.values, how="cross")  # Build the naive product
        for newColor in colorsShapeDict.keys():  # Drop zero rows
            if newColor in self.colors and newColor in core2.colors:
                preValues = preValues[
                    preValues[newColor + "_x"] == preValues[newColor + "_y"] & ~preValues[newColor + "_x"].astype(
                        str).str.contains(str(self.nanValue)) & ~preValues[newColor + "_y"].astype(str).str.contains(
                        str(core2.nanValue))].drop(newColor + "_y",
                                                   axis=1)
        contractedValues = preValues.rename(
            columns={col: col[:-2] for col in preValues.columns if col.endswith("_x") or col.endswith("_y")})
        contractedValues[self.valueColumn] = contractedValues[self.valueColumn] * contractedValues[
            self.valueColumn + "_sec"]
        contractedValues = contractedValues.drop(self.valueColumn + "_sec", axis=1)
        return PandasCore(values=contractedValues,
                          colors=list(colorsShapeDict.keys()),
                          shape=list(colorsShapeDict.values()),
                          valueColumn=self.valueColumn)

    def reduce_colors(self, newColors):
        ## Add correcting factors for trivial colors to be dropped
        self.values = self.values.reset_index()  # Unclear, in which cases this is necessary before the usage of loc
        for j in range(len(self.values)):
            self.values.loc[j, self.valueColumn] = np.prod([self.shape[k] for k, col in enumerate(self.colors) if
                                                            self.values.loc[
                                                                j, col] == self.nanValue and col not in newColors]) * \
                                                   self.values.loc[j, self.valueColumn]
        if len(newColors) == 0:
            self.values = pd.DataFrame({self.valueColumn: [self.values[self.valueColumn].sum()]})
        else:
            self.values = self.values.groupby(newColors)[self.valueColumn].sum().reset_index()

        self.shape = [self.shape[self.colors.index(color)] for i, color in enumerate(newColors) if color in newColors]
        self.colors = newColors

    def add_identical_slices(self):
        self.values = self.values.groupby(self.colors)[self.valueColumn].sum().reset_index()

    def slice_multiply(self, weight, sliceDict=dict()):
        """
        Cannot handle yet situation of nans in sliceDict
        """
        combined_condition = np.ones(self.values.shape[0], dtype=bool)
        for col, value in sliceDict.items():
            combined_condition &= (self.values[col] == value)
        self.values.loc[combined_condition, self.valueColumn] *= weight
        return self

    def get_slice(self, colorEvidenceDict={}):
        query = None
        for color in colorEvidenceDict:
            condition = (self.values[color] == colorEvidenceDict[color]) | (self.values[color] == self.nanValue)
            query = condition if query is None else query & condition
        newValues = self.values[query].drop(columns=colorEvidenceDict.keys())
        newColors = [color for color in self.colors if color not in colorEvidenceDict]
        newShape = [self.shape[i] for i, color in enumerate(self.colors) if color not in colorEvidenceDict]
        return PandasCore(values=newValues, colors=newColors, shape=newShape, name="Sliced_" + self.name)

    def reorder_colors(self, newColors):
        if set(self.colors) == set(newColors):
            self.colors = newColors
        else:
            raise ValueError("Reordering of Colors in Core {} not possible, since different!".format(self.name))

    def __add__(self, otherCore):
        otherCore.values = otherCore.values.rename(columns={otherCore.valueColumn: self.valueColumn})

        colorsShapeDict = {**{color: self.shape[i] for i, color in enumerate(self.colors)},
                           **{color: otherCore.shape[i] for i, color in enumerate(otherCore.colors)}}
        return PandasCore(values=pd.concat([self.values, otherCore.values], ignore_index=True),
                          colors=list(colorsShapeDict.keys()),
                          shape=list(colorsShapeDict.values()),
                          valueColumn=self.valueColumn)

    def __rmul__(self, scalar):
        self.values[self.valueColumn] = scalar * self.values[self.valueColumn]
        return self

    def enumerate_slices(self, enumerationColor="j"):
        self.values[enumerationColor] = [i for i in range(len(self.values))]
        self.colors = self.colors + [enumerationColor]
        self.shape = self.shape + [len(self.values)]


class PandasTermCore:

    def __init__(self, valueDf=None, startInterpretationDict=dict(), adjustWhileIterate=True):
        self.interpretationDict = startInterpretationDict
        if valueDf is not None:
            self.load(valueDf)

        self.adjustWhileIterate = adjustWhileIterate

    def load(self, valueDf, termVariables=None, adjustInterpretation=True, name="PandasDf"):
        if termVariables is None:
            self.termVariables = list(valueDf.columns)
            self.valueDf = valueDf
        else:
            self.termVariables = termVariables
            self.valueDf = valueDf[termVariables]
        for variable in self.termVariables:
            if variable not in self.interpretationDict:
                self.interpretationDict[variable] = []

        if adjustInterpretation:
            self.adjust_interpretationsDict()

        self.colors = [var + suf.terVarSuf for var in self.termVariables]
        self.shape = [len(self.interpretationDict[var]) for var in self.termVariables]
        self.name = name

    def adjust_interpretationsDict(self):
        """
        Necessary before shape is used
        """
        for j, row in self.valueDf.iterrows():
            for variable in self.termVariables:
                if str(row[variable]) not in self.interpretationDict[variable]:
                    self.interpretationDict[variable].append(str(row[variable]))

        self.shape = [len(self.interpretationDict[variable]) for variable in self.termVariables]
        self.adjustWhileIterate = False

    def __iter__(self):
        self.rowIterator = iter(self.valueDf.iterrows())
        return self

    def __next__(self):
        j, row = next(self.rowIterator)

        if self.adjustWhileIterate:
            for variable in self.termVariables:
                if str(row[variable]) not in self.interpretationDict[variable]:
                    self.interpretationDict[variable].append(str(row[variable]))

        return (1, {
            variable + suf.terVarSuf: self.interpretationDict[variable].index(str(row[variable]))
            for variable in self.termVariables})
