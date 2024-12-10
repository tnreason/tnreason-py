import numpy as np

defaultCoreType = "NumpyCore"


def get_core(coreType=None):
    if coreType is None:
        coreType = defaultCoreType
    if coreType == "NumpyCore":
        from tnreason.engine.workload_to_numpy import NumpyCore
        return NumpyCore
    elif coreType == "PolynomialCore":
        from tnreason.engine.polynomial_handling import PolynomialCore
        return PolynomialCore
    elif coreType == "PandasCore":
        from tnreason.engine.workload_to_pandas import PandasCore
        return PandasCore
    elif coreType == "HypertrieCore":
        from tnreason.engine.workload_to_tentris import HypertrieCore
        return HypertrieCore
    elif coreType == "TorchCore":
        from tnreason.engine.workload_to_torch import TorchCore
        return TorchCore
    elif coreType == "TensorFlowCore":
        from tnreason.engine.workload_to_tensorflow import TensorFlowCore
        return TensorFlowCore
    else:
        raise ValueError("Core Type {} not supported.".format(coreType))


def create_tensor_encoding(inshape, incolors, function, coreType=None, name="Encoding"):
    if coreType is None:
        coreType = defaultCoreType
    return create_from_slice_iterator(inshape, incolors,
                                      sliceIterator=[
                                          (function(*idx), {color: idx[i] for i, color in enumerate(incolors)}) for
                                          idx in np.ndindex(*inshape)],
                                      coreType=coreType, name=name)


def create_random_core(name, shape, colors,
                       randomEngine="NumpyUniform"):  # Works only for numpy cores! (do not have a random engine else)
    from tnreason.engine.workload_to_numpy import np_random_core
    return np_random_core(shape, colors, randomEngine, name)


def create_relational_encoding(inshape, outshape, incolors, outcolors, function, coreType=None,
                               name="Encoding"):
    """
    Creates relational encoding of a function as a single core.
    The function has to be a map from the indices in inshape to the indices in outshape.
    """
    if coreType is None:
        coreType = defaultCoreType
    return create_from_slice_iterator(inshape + outshape, incolors + outcolors,
                                      sliceIterator=[(1, {**{color: idx[i] for i, color in enumerate(incolors)},
                                                          **{color: int(function(*idx)[i]) for i, color in
                                                             enumerate(outcolors)}}) for idx in np.ndindex(*inshape)],
                                      coreType=coreType, name=name)


def coordinate_slice_iterator(inshape, incolors, coordFunction):
    return [(coordFunction(*idx), {color: idx[i] for i, color in enumerate(incolors)}) for idx in np.ndindex(*inshape)]


def create_from_slice_iterator(shape, colors, sliceIterator, coreType=defaultCoreType, name="Iterator"):
    core = get_core(coreType)(values=None, colors=colors, name=name, shape=shape)
    for value, sliceDict in sliceIterator:
        core[sliceDict] = value
    return core


def convert(inCore, outCoreType=None):
    if outCoreType is None:
        outCoreType = defaultCoreType
    return create_from_slice_iterator(inCore.shape, inCore.colors, iter(inCore), coreType=outCoreType)


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
    Creates relational encoding of a function as a tensor network, where the output axis are splitted according to the partionDict.
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
