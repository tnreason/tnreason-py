import numpy as np

from tnreason import engine


def create_relational_encoding(inshape, outshape, incolors, outcolors, function, coreType=None,
                               name="Encoding"):
    """
    Creates relational representation of a function as a single core.
    The function has to be a map from the indices in inshape to the indices in outshape.
    """
    return engine.create_from_slice_iterator(inshape + outshape, incolors + outcolors,
                                      sliceIterator=[(1, {**{color: idx[i] for i, color in enumerate(incolors)},
                                                          **{color: int(function(*idx)[i]) for i, color in
                                                             enumerate(outcolors)}}) for idx in np.ndindex(*inshape)],
                                      coreType=coreType, name=name)


def core_to_relational_encoding(core, headColor, outCoreType=None):
    imageValues = get_image(core, core.shape)
    return create_relational_encoding(inshape=core.shape, outshape=[len(imageValues)], incolors=core.colors,
                                      outcolors=[headColor], function=lambda *args: [imageValues.index(core[args])],
                                      coreType=outCoreType), imageValues


def get_image(core, inShape, imageValues=[float(0), float(1)]):
    import numpy as np
    for indices in np.ndindex(tuple(inShape)):
        coordinate = float(core[indices])
        if coordinate not in imageValues:
            imageValues.append(coordinate)
    return imageValues


def create_partitioned_relational_encoding(inshape, outshape, incolors, outcolors, function, coreType=None,
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


def create_interpretation_vector(color, coreType=None, name=None, interImage=[0,1]):
    """
    Creates the vector interpretation of a term variable color, where interImage specifies the interpretation
    """
    return engine.create_from_slice_iterator(
        shape=[len(interImage)], colors=[color],
        sliceIterator=[(interImage[i], {color: i}) for i in range(len(interImage))],
        coreType=coreType, name=name
    )
