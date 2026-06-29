def add_doVariable(doColor, headColor, doDim, sliceIterator):
    """
    Adds the do Variable to the sliceIterator describing a conditional distribution
    """
    return [(val, {**pos, doColor: doDim}) for (val, pos) in sliceIterator] + [(1, {headColor: i, doColor : i }) for i in range(doDim)]