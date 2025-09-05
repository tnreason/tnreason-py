import numpy as np

from tnreason import engine

interpretationCorePre = "_i"


def atomic_image_enumeration(function, domainIterator):
    """
    Treats the function domain and range as atomic state sets, i.e. as enumerated finite sets.
    """
    imageValues = []
    for idx in domainIterator:
        value = function(*idx)
        if value not in imageValues:
            imageValues.append(value)
    return imageValues


## Unused so far, since typically the statesFunction is encoded in decomposed form
def encode_statesFunction(statesFunction, inshape, incolors, outcolor, coreType=None, coreName="Encoding",
                          imageList=None):
    """
    Atomic encoding of the head!
    statesFunction : function from \bigtimes_{\catenumeratorin}[\catdimof{\catenumerator}] to set of values (storable in a list)
    inshape : list [catdimof{\catenumerator} : \catenumeratorin]
    """
    if imageList is None:
        imageList = atomic_image_enumeration(statesFunction, domainIterator=np.ndindex(*inshape))
    statesToIndexFunction = lambda *args: [imageList.index(statesFunction(*args))]
    return create_basis_encoding_from_lambda(inshape=inshape, outshape=[len(imageList)], incolors=incolors,
                                                  outcolors=[outcolor], indicesToIndicesFunction=statesToIndexFunction,
                                                  coreType=coreType,
                                                  name=coreName), \
        create_interpretation_vector(outcolor, coreType=coreType, name=outcolor + "_i")


def create_basis_encoding_from_lambda(inshape, outshape, incolors, outcolors, indicesToIndicesFunction,
                                           coreType=None,
                                           name="Encoding"):
    """
    Creates relational representation of a function as a single core.
    The function has to be a map from the indices in inshape to the indices in outshape.
    """
    return engine.create_from_slice_iterator(outshape + inshape, outcolors + incolors,
                                             sliceIterator=[(1, {**{color: idx[i] for i, color in enumerate(incolors)},
                                                                 **{color: int(indicesToIndicesFunction(*idx)[i]) for
                                                                    i, color in
                                                                    enumerate(outcolors)}}) for idx in
                                                            np.ndindex(*inshape)],
                                             coreType=coreType, name=name)


## Unused so far: Building the relational encoding of a core itself
def core_to_basis_encoding(core, headColor, outCoreType=None):
    imageList = atomic_image_enumeration(
        function=lambda *args: core[{color: args[i] for i, color in enumerate(core.colors)}],
        domainIterator=np.ndindex(*core.shape))
    return create_basis_encoding_from_lambda(inshape=core.shape, outshape=[len(imageList)], incolors=core.colors,
                                                  outcolors=[headColor],
                                                  indicesToIndicesFunction=lambda *args: [imageList.index(core[args])],
                                                  coreType=outCoreType), create_interpretation_vector(color=headColor,
                                                                                                      interImage=imageList)


def get_image(core, inShape, imageValues=[float(0), float(1)]):
    import numpy as np
    for indices in np.ndindex(tuple(inShape)):
        coordinate = float(core[indices])
        if coordinate not in imageValues:
            imageValues.append(coordinate)
    return imageValues


def create_partitioned_basis_encoding(inshape, outshape, incolors, outcolors, function, coreType=None,
                                           partitionDict=None, nameSuffix="_encodingCore"):
    """
    Creates relational representation of a function as a tensor network, where the output axis are splitted according to the partionDict.
    """
    if partitionDict is None:
        partitionDict = {color: [color] for color in outcolors}
    return {parKey + nameSuffix:
                create_basis_encoding_from_lambda(inshape=inshape,
                                                       outshape=[outshape[outcolors.index(c)] for c in
                                                                 partitionDict[parKey]],
                                                       incolors=incolors,
                                                       outcolors=partitionDict[parKey],
                                                       indicesToIndicesFunction=lambda x: [
                                                           function(x)[outcolors.index(c)] for c in
                                                           partitionDict[parKey]],
                                                       coreType=coreType,
                                                       name=parKey + nameSuffix)
            for parKey in partitionDict}


def create_interpretation_vector(color, coreType=None, name=None, interImage=[0, 1]):
    """
    Creates the vector interpretation of a term variable color, where interImage specifies the interpretation
    """
    return engine.create_from_slice_iterator(
        shape=[len(interImage)], colors=[color],
        sliceIterator=[(interImage[i], {color: i}) for i in range(len(interImage))],
        coreType=coreType, name=name
    )


if __name__ == "__main__":
    # Example usage
    def example_function(x, y):
        return x + y


    inshape = [2, 2]
    incolors = ['x', 'y']
    outcolor = 'z'
    core, interpretation_vector = encode_statesFunction(example_function, inshape, incolors, outcolor)

    relCore, intCore = core_to_basis_encoding(core, headColor="fun")
    assert intCore[{"fun": 0}] in {0, 1} and intCore[{"fun": 1}] in {0, 1}, "Interpretation vector should be binary!"

    print("Core:", core)
    print("Interpretation Vector:", interpretation_vector)
    print("Relational encoding of Core:", core)
    print("Interpretation Vector:", interpretation_vector)
