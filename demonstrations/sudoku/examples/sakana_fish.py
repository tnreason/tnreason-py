"""
We here provide templates for creating the cores needed for implementing the constraints on the Sakana Fish Sudoku puzzle.

We exploit either the more fundamental operations of tnreason.engine, or more abstract functionality from tnreason.representation.

To be more precise:
- Red line: We instantiate the hidden variable that indicates whether a position is odd or even, and we create the constraints by xor on the hidden variables.
- White dot: We create the constraint that two positions differ by one, based .
- Black dot: We create the constraint that one position is double the other.
"""

from tnreason import engine


### Directly on the slice iterator: Using engine.engine
def prepare_odd_indicator_core(posVar, oddVar, sudokuNum=3):
    """
    Prepares a hidden variable oddVar that indicates whether the position variable posVar is odd or even.
    """
    return engine.create_from_slice_iterator(
        shape=[sudokuNum ** 2, 2],
        colors=[posVar, oddVar],
        sliceIterator=[(1, {posVar: val, oddVar: (val + 1) % 2}) for val in range(sudokuNum ** 2)]
    )


def white_dot_constraint(posVar1, posVar2, sudokuNum=3):
    """
    Both positions have the same parity
    """
    return engine.create_from_slice_iterator(
        shape=[sudokuNum ** 2, sudokuNum ** 2],
        colors=[posVar1, posVar2],
        sliceIterator=[(1, {posVar1: val1, posVar2: val2}) for val1 in range(sudokuNum ** 2) for val2 in
                       range(sudokuNum ** 2) if abs(val1 - val2) == 1])


def black_dot_constraint(posVar1, posVar2, sudokuNum=3):
    """
    Both positions have the same parity
    """
    return engine.create_from_slice_iterator(
        shape=[sudokuNum ** 2, sudokuNum ** 2],
        colors=[posVar1, posVar2],
        sliceIterator=[(1, {posVar1: val1, posVar2: val2}) for val1 in range(sudokuNum ** 2) for val2 in
                       range(sudokuNum ** 2) if (val1 + 1) == 2 * (val2 + 1) or (val2 + 1) == 2 * (val1 + 1)])


def red_line_constraint(oddInd1, oddInd2):
    """
    Exactly one of the neighbored positions is odd
    """
    return engine.create_from_slice_iterator(
        shape=[2, 2],
        colors=[oddInd1, oddInd2],
        sliceIterator=[(1, {oddInd1: 0, oddInd2: 1}),
                       (1, {oddInd1: 1, oddInd2: 0})])


## Using the abstraction of tnreason.representation
from tnreason import representation


### Using the basis encoding in engine.representation
def benc_prepare_odd_indicator_core(posVar, oddVar, sudokuNum=3):
    return representation.create_basis_encoding_from_lambda(
        inshape=[sudokuNum ** 2],
        incolors=[posVar],
        outshape=[2],
        outcolors=[oddVar],
        indicesToIndicesFunction=lambda x: [(x + 1) % 2])


def cenc_red_line_constraint(oddInd1, oddInd2):
    return representation.create_tensor_encoding(inshape=[2, 2],
                                                 incolors=[oddInd1, oddInd2],
                                                 function=lambda x, y: int(
                                                     not x == y))


def cenc_white_dot_constraint(posVar1, posVar2, sudokuNum=3):
    """
    Both positions have the same parity
    """
    return representation.create_tensor_encoding(inshape=[sudokuNum ** 2, sudokuNum ** 2], incolors=[posVar1, posVar2],
                                                 function=lambda val1, val2: abs(val1 - val2) == 1)


def cenc_black_dot_constraint(posVar1, posVar2, sudokuNum=3):
    """
    Both positions have the same parity
    """
    return representation.create_tensor_encoding(inshape=[sudokuNum ** 2, sudokuNum ** 2], incolors=[posVar1, posVar2],
                                                 function=lambda val1, val2: (val1 + 1) == 2 * (val2 + 1) or (
                                                         val2 + 1) == 2 * (val1 + 1))


def cenc_red_line_constraint(oddInd1, oddInd2):
    """
    Exactly one of the neighbored positions is odd
    """
    return representation.create_tensor_encoding(inshape=[2, 2], incolors=[oddInd1, oddInd2],
                                                 function=lambda o1, o2: o1 ^ o2)


if __name__ == "__main__":
    assert prepare_odd_indicator_core("pos", "odd")[{"pos": 3, "odd": 0}] == 1
    assert prepare_odd_indicator_core("pos", "odd")[{"pos": 3, "odd": 1}] == 0
    assert benc_prepare_odd_indicator_core("pos", "odd")[{"pos": 3, "odd": 0}] == 1
    assert benc_prepare_odd_indicator_core("pos", "odd")[{"pos": 3, "odd": 1}] == 0

    assert red_line_constraint("o1", "o2")[{"o2": 1, "o1": 0}] == 1
    assert red_line_constraint("o1", "o2")[{"o2": 1, "o1": 1}] == 0
    assert cenc_red_line_constraint("o1", "o2")[{"o2": 1, "o1": 0}] == 1
    assert cenc_red_line_constraint("o1", "o2")[{"o2": 1, "o1": 1}] == 0

    assert white_dot_constraint("p1", "p2")[{"p1": 1, "p2": 0}] == 1
    assert white_dot_constraint("p1", "p2")[{"p1": 5, "p2": 6}] == 1
    assert cenc_white_dot_constraint("p1", "p2")[{"p1": 1, "p2": 0}] == 1
    assert cenc_white_dot_constraint("p1", "p2")[{"p1": 5, "p2": 6}] == 1

    assert black_dot_constraint("p1", "p2")[{"p1": 5, "p2": 6}] == 0
    assert black_dot_constraint("p1", "p2")[{"p1": 2, "p2": 5}] == 1
    assert cenc_black_dot_constraint("p1", "p2")[{"p1": 5, "p2": 6}] == 0
    assert cenc_black_dot_constraint("p1", "p2")[{"p1": 2, "p2": 5}] == 1
