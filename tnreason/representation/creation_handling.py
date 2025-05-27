import math

from tnreason import engine

import numpy as np

from tnreason.representation import suffixes as suf

defaultCoreType = "NumpyCore"

def create_tensor_encoding(inshape, incolors, function, coreType=None, name="Encoding"):
    if coreType is None:
        coreType = defaultCoreType
    return engine.create_from_slice_iterator(inshape, incolors,
                                      sliceIterator=[
                                          (function(*idx), {color: idx[i] for i, color in enumerate(incolors)}) for
                                          idx in np.ndindex(*inshape)],
                                      coreType=coreType, name=name)





def create_relational_encoding(inshape, outshape, incolors, outcolors, function, coreType=None,
                               name="Encoding"):
    """
    Creates relational representation of a function as a single core.
    The function has to be a map from the indices in inshape to the indices in outshape.
    """
    if coreType is None:
        coreType = defaultCoreType
    return engine.create_from_slice_iterator(inshape + outshape, incolors + outcolors,
                                      sliceIterator=[(1, {**{color: idx[i] for i, color in enumerate(incolors)},
                                                          **{color: int(function(*idx)[i]) for i, color in
                                                             enumerate(outcolors)}}) for idx in np.ndindex(*inshape)],
                                      coreType=coreType, name=name)


def coordinate_slice_iterator(inshape, incolors, coordFunction):
    return [(coordFunction(*idx), {color: idx[i] for i, color in enumerate(incolors)}) for idx in np.ndindex(*inshape)]


def get_image(core, inShape, imageValues=[float(0), float(1)]):
    import numpy as np
    for indices in np.ndindex(tuple(inShape)):
        coordinate = float(core[indices])
        if coordinate not in imageValues:
            imageValues.append(coordinate)
    return imageValues


def core_to_relational_encoding(core, headColor, outCoreType=None):
    imageValues = get_image(core, core.shape)
    return create_relational_encoding(inshape=core.shape, outshape=[len(imageValues)], incolors=core.colors,
                                      outcolors=[headColor], function=lambda *args: [imageValues.index(core[args])],
                                      coreType=outCoreType), imageValues


def reduce_function(function, coordinates):
    return lambda x: [function(x)[coordinate] for coordinate in coordinates]


def create_partitioned_relational_encoding(inshape, outshape, incolors, outcolors, function, coreType=defaultCoreType,
                                           partitionDict=None, nameSuffix="_encodingCore"):
    """
    Creates relational representation of a function as a tensor network, where the output axis are splitted according to the partionDict.
    """
    if partitionDict is None:
        partitionDict = {color: [color] for color in outcolors}
    return {parKey + nameSuffix:
                create_relational_encoding(inshape=inshape,
                                           outshape=[outshape[outcolors.index(c)] for c in partitionDict[parKey]],
                                           incolors=incolors,
                                           outcolors=partitionDict[parKey],
                                           function=lambda x: [function(x)[outcolors.index(c)] for c in
                                                               partitionDict[parKey]],
                                           coreType=coreType,
                                           name=parKey + nameSuffix)
            for parKey in partitionDict}


## Coordinate Calculus -> Generalizing Numpy
def coordinatewise_transform(coreList, rDrFunction, outCoreType=None, outName="Transformed"):
    """
    Computed the coordiantewise transform of tensors
    * coreList: List of d tensor cores of same shape and colors
    * rDrFunction: Function from \mathbb{R}^d to \mathbb{R}, computing the coordinate of the output core
    """
    newCore = engine.get_core(coreType=outCoreType)(shape=coreList[0].shape, colors=coreList[0].colors, name=outName)
    for index in np.ndindex(coreList[0].shape):
        newCore[{color: index[k] for k, color in enumerate(coreList[0].colors)}] = rDrFunction(
            *[core[index] for core in coreList])
    return newCore

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


def create_trivial_core(name, shape, colors, coreType=None):
    return create_tensor_encoding(inshape=shape, incolors=colors, function=lambda *args: 1, coreType=coreType,
                                         name=name)

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
