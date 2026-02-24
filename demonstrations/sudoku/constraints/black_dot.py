from tnreason import engine

def black_dot_constraint(posVar1, posVar2, sudokuNum=3):
    """
    Both positions have the same parity
    """
    return engine.create_from_slice_iterator(
        shape=[sudokuNum ** 2, sudokuNum ** 2],
        colors=[posVar1, posVar2],
        sliceIterator=[(1, {posVar1: val1, posVar2: val2}) for val1 in range(sudokuNum ** 2) for val2 in
                       range(sudokuNum ** 2) if (val1 + 1) == 2 * (val2 + 1) or (val2 + 1) == 2 * (val1 + 1)])


## Using the abstraction of tnreason.representation
from tnreason import representation


def cenc_black_dot_constraint(posVar1, posVar2, sudokuNum=3):
    """
    Both positions have the same parity
    """
    return representation.create_tensor_encoding(inshape=[sudokuNum ** 2, sudokuNum ** 2], incolors=[posVar1, posVar2],
                                                 function=lambda val1, val2: (val1 + 1) == 2 * (val2 + 1) or (
                                                         val2 + 1) == 2 * (val1 + 1))

if __name__ == "__main__":
    assert black_dot_constraint("p1", "p2")[{"p1": 5, "p2": 6}] == 0
    assert black_dot_constraint("p1", "p2")[{"p1": 2, "p2": 5}] == 1
    assert cenc_black_dot_constraint("p1", "p2")[{"p1": 5, "p2": 6}] == 0
    assert cenc_black_dot_constraint("p1", "p2")[{"p1": 2, "p2": 5}] == 1