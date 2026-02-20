import numpy as np

from tnreason import engine


def coordinatewise_transform(coreList, rDrFunction, outCoreType=None, outName="Transformed"):
    """
    Computed the coordinatewise transform of tensors
    * coreList: List of d tensor cores of same shape and colors
    * rDrFunction: Function from \mathbb{R}^d to \mathbb{R}, computing the coordinate of the output core
    """
    return engine.create_from_slice_iterator(shape=coreList[0].shape, colors=coreList[0].colors,
                                             sliceIterator=[(rDrFunction(*[core[index] for core in coreList]),
                                                             {color: index[i] for i, color in
                                                              enumerate(coreList[0].colors)})
                                                            for index in np.ndindex(*coreList[0].shape)],
                                             coreType=outCoreType, name=outName)


def create_tensor_encoding(inshape, incolors, function, coreType=None, name="Encoding"):
    """
    Uses a dense sliceIterator, more efficient (when sparsity capturing coreType) in case of trivial, vanishing or basis core below
    """
    return engine.create_from_slice_iterator(inshape, incolors,
                                             sliceIterator=[
                                                 (function(*idx), {color: idx[i] for i, color in enumerate(incolors)})
                                                 for idx in np.ndindex(*inshape)],
                                             coreType=coreType, name=name)


## Special Tensor Encodings with sparser sliceIterators :

def create_vanishing_core(colors, shape, name, coreType=None):
    return engine.create_from_slice_iterator(shape=shape, colors=colors,
                                             sliceIterator=[],
                                             coreType=coreType, name=name)


def create_trivial_core(colors, shape, name, coreType=None):
    return engine.create_from_slice_iterator(shape=shape, colors=colors,
                                             sliceIterator=[(1, dict())],
                                             coreType=coreType, name=name)


def create_basis_core(name, shape, colors, numberTuple, coreType=None):
    if isinstance(numberTuple, tuple) or isinstance(numberTuple, list):
        numberTuple = tuple([int(number) for number in numberTuple])
    else:  # Dealing with np.int, Booleans, Floats
        numberTuple = tuple([int(numberTuple)])
    return engine.create_from_slice_iterator(shape=shape, colors=colors,
                                             sliceIterator=[
                                                 (1, {color: numberTuple[i] for i, color in enumerate(colors)})],
                                             coreType=coreType, name=name)