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
                                                 function=lambda x, y: int(not x == y))


def cenc_red_line_constraint(oddInd1, oddInd2):
    """
    Exactly one of the neighbored positions is odd
    """
    return representation.create_tensor_encoding(inshape=[2, 2], incolors=[oddInd1, oddInd2],
                                                 function=lambda o1, o2: o1 ^ o2)