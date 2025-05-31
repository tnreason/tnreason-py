import numpy as np

from tnreason import engine


def coordinatewise_transform(coreList, rDrFunction, outCoreType=None, outName="Transformed"):
    """
    Computed the coordiantewise transform of tensors
    * coreList: List of d tensor cores of same shape and colors
    * rDrFunction: Function from \mathbb{R}^d to \mathbb{R}, computing the coordinate of the output core
    """
    return engine.create_from_slice_iterator(shape=coreList[0].shape, colors=coreList[0].colors,
                                                       sliceIterator=[(
                                                                      rDrFunction(*[core[index] for core in coreList]),
                                                                      {color: index[i] for i, color in
                                                                       enumerate(coreList[0].colors)}) for index in
                                                                      np.ndindex(*coreList[0].shape)]
                                                       , coreType=outCoreType, name=outName)
    # newCore = engine.get_core(coreType=outCoreType)(shape=coreList[0].shape, colors=coreList[0].colors, name=outName)
    # for index in np.ndindex(*coreList[0].shape):  ## can be abbreviated as function for create_from_slice_iterator!
    #     newCore[{color: index[k] for k, color in enumerate(coreList[0].colors)}] = rDrFunction(
    #         *[core[index] for core in coreList])
    # return newCore


def create_tensor_encoding(inshape, incolors, function, coreType=None, name="Encoding"):
    # if coreType is None:
    #    coreType = defaultCoreType
    return engine.create_from_slice_iterator(inshape, incolors,
                                             sliceIterator=[
                                                 (function(*idx), {color: idx[i] for i, color in enumerate(incolors)})
                                                 for
                                                 idx in np.ndindex(*inshape)],
                                             coreType=coreType, name=name)
