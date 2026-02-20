import numpy as np


class TensorCore:
    coreType = "Base"

    def __init__(self, colors, name, shape):
        self.colors = colors
        self.name = name

        if shape is None:
            self.shape = [2 for _ in self.colors]
        else:
            self.shape = shape

        if len(self.colors) != len(self.shape):
            raise ValueError("Number of Colors does not match the Value Shape in Core {}!".format(name))
        if len(self.colors) != len(set(self.colors)):
            raise ValueError("There are duplicate colors in the colors {} of Core {}!".format(colors, name))

    def __str__(self):
        return "## Core " + self.name + " ##\nType: " + str(self.coreType) + " ##\nShape: " + str(
            self.shape) + "\nColors: " + str(self.colors)

    def __eq__(self, other):
        """
        Cores are considered equal, when storing the same tensor. Thus, the equality check is ignorant about the core type.
        """
        if sorted(self.shape) != sorted(other.shape):  # Check whether mismatch in shape or dimDict
            return False

        for i, color in enumerate(self.colors):
            if color in other.colors:
                if not self.shape[i] == other.shape[other.colors.index(color)]:
                    return False
            else:
                return False
        #lse:  # Then check their values
        for index in np.ndindex(*self.shape):
                colorPosDict = {color: index[i] for i, color in enumerate(self.colors)}
                if self[colorPosDict] != other[colorPosDict]:
                    return False
        return True
