import math

from tnreason import engine

from tnreason.representation import suffixes as suf#, create_tensor_encoding

import numpy as np


# Now in coordiante calculus!
def create_tensor_encoding(inshape, incolors, function, coreType=None, name="Encoding"):
    return engine.create_from_slice_iterator(inshape, incolors,
                                      sliceIterator=[
                                          (function(*idx), {color: idx[i] for i, color in enumerate(incolors)}) for
                                          idx in np.ndindex(*inshape)],
                                      coreType=coreType, name=name)

def create_trivial_cores(rawKeys, shapeDict=None, suffix="", coreType=None):
    """
    Creates dictionary of trivial cores with coordinate 1, which act as neutral placeholders in contractions.
        * rawKeys: List of raw keys (added by suffix)
        * shapeDict: Dictionary of shapes of the trivial core to each rawKey, default: 2
    """
    if shapeDict is None:
        shapeDict = {key: 2 for key in rawKeys}
    return {key + suffix: create_trivial_core(key + suffix, shapeDict[key], [key], coreType=coreType) for key in
            rawKeys}

## To do: integrate
def create_activation_vector(color, canParam=0, supportConstraint=None, coreType=None, name=None, interImage=[0,1]):
    """
    Creates a generic activation core to a feature of an exponential statistic
        * supportConstraint: Support of the activation vector
        * canParam: Canonical parameter of the feature to the activation core
        * interImage: Image of the interpretation function to the computed variable, which enumerates with [len]. By default: boolean [0,1].
    """
    if name is None:
        name = color + suf.actCoreSuf
    if supportConstraint is None:
        interFunction = lambda x: math.exp(canParam * x)
    else:
        interFunction = lambda x: math.exp(canParam * x) * int(x in supportConstraint)
    return engine.create_from_slice_iterator(shape=[len(interImage)], colors=[color],
                                      sliceIterator=[
                                          (interFunction(entry), {color: i}) for i, entry in enumerate(interImage)],
                                      coreType=coreType, name=name)


## Special cases of activation cores
def create_trivial_core(name, shape, colors, coreType=None):
    return create_tensor_encoding(inshape=shape, incolors=colors, function=lambda *args: 1, coreType=coreType,
                                  name=name)

def create_vanishing_core(colors, shape, coreType=None, name=None):
    return engine.create_from_slice_iterator(shape=shape, colors=colors, sliceIterator=[], coreType=coreType, name=name)


def create_basis_core(name, shape, colors, numberTuple, coreType=None):
    if isinstance(numberTuple, tuple) or isinstance(numberTuple, list):
        numberTuple = tuple([int(number) for number in numberTuple])
    else:  # Dealing with np.int, Booleans, Floats
        numberTuple = tuple([int(numberTuple)])
    return create_tensor_encoding(inshape=shape, incolors=colors,
                                  function=lambda *args: int(args == numberTuple), coreType=coreType, name=name)


def create_boolean_head(color, headType, weight=None, coreType=None, name=None):
    """
    Created the head core to a boolean variable (with dimension 2)
    CHANGE: Expfactor and uncertainty not boolean!
    """
    if headType == "truthEvaluation": # tBasis
        headFunction = lambda x: x
    elif headType == "falseEvaluation": # fBasis
        headFunction = lambda x: 1 - x
    elif headType == "expFactor":
        headFunction = lambda x: math.exp(weight * x)
    elif headType == "uncertaintyAsWeight":
        headFunction = lambda x: (1 - x) * (1 - weight) + x * weight
    else:
        raise ValueError("Headtype {} not understood!".format(headType))
    if name is None:
        name = color + suf.actCoreSuf
    return {name: create_tensor_encoding([2], [color], headFunction, coreType=coreType,
                                         name=name)}