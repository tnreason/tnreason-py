from tnreason import engine

import numpy as np

aliases = {
    "xor": "neq",
    "lpas": "pas0",
    "id": "pas0",
    "not": "npas0"
}


def get_selection_augmented_boolean_computation_core(functionNames, inColors, outColor, selectionColor, coreType, name):
    return engine.create_from_slice_iterator(shape=[2 for _ in range(len(inColors) + 1)],
                                             colors=[outColor] + inColors,
                                             sliceIterator=get_selection_augmented_iterator(functionNames, inColors,
                                                                                            outColor, selectionColor),
                                             coreType=coreType, name=name)


def get_selection_augmented_iterator(functionNames, inColors, outColor, selectionColor):
    """
    Naive encoding of activation selection tensors, by summation over the selectable relational encodings
    """
    return [(val, {**pos, selectionColor: i}) for i, functionName in enumerate(functionNames) for (val, pos) in
            get_boolean_relational_value_iterator(functionName, inColors, outColor)]


def get_boolean_computation_core(functionName, inColors, outColor, coreType, name):
    return engine.create_from_slice_iterator(shape=[2 for _ in range(len(inColors) + 1)],
                                             colors=[outColor] + inColors,
                                             sliceIterator=get_boolean_relational_value_iterator(functionName, inColors,
                                                                                                 outColor),
                                             coreType=coreType, name=name)


def get_boolean_relational_value_iterator(functionName, inColors, outColor):
    if functionName in aliases:
        functionName = aliases[functionName]

    if functionName.startswith("n"):
        """
        Then flip the head variable (effectively a contraction with the not gate.
        """
        return [(val,
                 {outColor: int(1 - pos[outColor]), **{inColor: pos[inColor] for inColor in pos if inColor != outColor}}
                 ) for (val, pos) in
                get_boolean_relational_value_iterator(functionName[1:], inColors=inColors, outColor=outColor)]

    if functionName.startswith("pas"):
        pos = int(functionName[3:])
        assert pos < len(inColors)
        return [(1, {outColor: 0, inColors[pos]: 0}),
                (1, {outColor: 1, inColors[pos]: 1})]

    """
    Provides the value iterator to boolean functions for relational encodings
    """
    if functionName in ["and", "or", "eq", "imp", "rpas"]:
        """
        Then a specific sparse function with logical interpretation
        """
        if functionName == "and":  # Logical and
            return [(1, {outColor: 0}),
                    (-1, {outColor: 0, **{inColor: 1 for inColor in inColors}}),
                    (1, {outColor: 1, **{inColor: 1 for inColor in inColors}})]
        elif functionName == "or":  # Logical or
            return [(1, {outColor: 1}),
                    (-1, {outColor: 1, **{inColor: 0 for inColor in inColors}}),
                    (1, {outColor: 0, **{inColor: 0 for inColor in inColors}})]
        elif functionName == "eq":
            return [(1, {outColor: 0}),
                    (-1, {outColor: 0, **{inColor: 1 for inColor in inColors}}),
                    (-1, {outColor: 0, **{inColor: 0 for inColor in inColors}}),
                    (1, {outColor: 1, **{inColor: 1 for inColor in inColors}}),
                    (1, {outColor: 1, **{inColor: 0 for inColor in inColors}})]
        elif functionName == "imp":
            return [(1, {outColor: 1}),
                    (-1, {outColor: 1, **{inColor: 1 for inColor in inColors[:-1]}, inColors[-1]: 0}),
                    (1, {outColor: 0, **{inColor: 1 for inColor in inColors[:-1]}, inColors[-1]: 0})]

        elif functionName == "rpas":
            return get_boolean_relational_value_iterator("pas" + str(len(inColors) - 1), inColors=inColors,
                                                         outColor=outColor)

    ## Then a Wolfram number
    try:
        int(functionName)
    except:
        raise ValueError("Function {} is not a Wolfram number".format(functionName))

    binDigits = bin(int(functionName))[2:]
    order = len(inColors)
    if len(binDigits) != 2 ** order:
        binDigits = "0" * (2 ** order - len(binDigits)) + binDigits
    return [(1, {outColor: int(binDigits[2 ** order - 1 - int("".join(map(str, indices)), 2)]),
                 **{inColor: indices[i] for i, inColor in enumerate(inColors)}}) for indices in
            np.ndindex(*[2 for _ in range(order)])]


if __name__ == "__main__":

    print(get_boolean_relational_value_iterator("123", inColors=["asf", "blub", "fun"], outColor="h"))

    print([el for el in
           get_boolean_relational_value_iterator(functionName="nand", inColors=["asf", "blub", "fun"], outColor="h")])

    from tnreason import engine

    core = engine.get_core()(colors=["asf", "blub", "fun", "h"])
    for el in get_boolean_relational_value_iterator("pas2", ["asf", "blub", "fun"], "h"):
        core[el[1]] = el[0]
        print(el)

    print(get_boolean_relational_value_iterator("not", ["a", "b"], "hat"))
