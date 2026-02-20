from tnreason import engine

def white_dot_constraint(posVar1, posVar2, sudokuNum=3):
    """
    Both positions have the same parity
    """
    return engine.create_from_slice_iterator(
        shape=[sudokuNum ** 2, sudokuNum ** 2],
        colors=[posVar1, posVar2],
        sliceIterator=[(1, {posVar1: val1, posVar2: val2}) for val1 in range(sudokuNum ** 2) for val2 in
                       range(sudokuNum ** 2) if abs(val1 - val2) == 1])


## Using the abstraction of tnreason.representation
from tnreason import representation


def cenc_white_dot_constraint(posVar1, posVar2, sudokuNum=3):
    """
    Both positions have the same parity
    """
    return representation.create_tensor_encoding(inshape=[sudokuNum ** 2, sudokuNum ** 2], incolors=[posVar1, posVar2],
                                                 function=lambda val1, val2: abs(val1 - val2) == 1)