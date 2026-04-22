"""
Exact cover problems generalize (standard) Sudoku and n-queens

Problems are specified by dictionaries
- keys: items (str)
- values: lists of sets including an item

Here: Polyomino tiling as an exact cover problem

"""

from tnreason import engine


def get_constraintNetwork(constraintDict):
    return {f"{constraintKey}_{atomKey}": engine.create_from_slice_iterator(
        shape=[2, len(constraintDict[constraintKey])], colors=[atomKey, constraintKey],
        sliceIterator=[(1, {atomKey: 0}), (-1, {atomKey: 0, constraintKey: i}), (1, {atomKey: 1, constraintKey: i})]
    ) for constraintKey in constraintDict for i, atomKey in enumerate(constraintDict[constraintKey])}


## For polyomino tiling

def turn_into_constraintDict(setDict):
    constraintDict = dict()
    for setKey in setDict:
        for (xPos, yPos) in setDict[setKey]:
            if f"{xPos}_{yPos}" not in constraintDict:
                constraintDict[f"{xPos}_{yPos}"] = [setKey]
            else:
                constraintDict[f"{xPos}_{yPos}"].append(setKey)
    return constraintDict


def get_all_sets(xLength, yLength, polyominoeDict):
    setDict = dict()
    for polyominoKey in polyominoeDict:
        for (xShift, yShift, shifted) in get_polyomino_sets(xLength, yLength, polyominoeDict[polyominoKey]):
            setDict[f"{polyominoKey}_{xShift}_{yShift}"] = shifted
    return setDict


def get_polyomino_sets(xLength, yLength, polyominoes):
    included = []
    for xShift in range(xLength):
        for yShift in range(yLength):
            shifted = shift_polyominoe(polyominoes, xShift, yShift)
            if polyominoe_fits(xLength, yLength, shifted):
                included.append((xShift, yShift, shifted))
    return included


def shift_polyominoe(polyominoes, xShift, yShift):
    return [(xPos + xShift, yPos + yShift) for (xPos, yPos) in polyominoes]


def polyominoe_fits(xLength, yLength, polyominoes):
    return all([xPos < xLength and xPos >= 0 and yPos < yLength and yPos >= 0 for (xPos, yPos) in polyominoes])


if __name__ == "__main__":
    ## Check number of assignments of vertical domino in a square
    xLength = 4
    yLength = 8
    assert len(
        get_polyomino_sets(xLength=xLength, yLength=yLength, polyominoes=[(0, 0), (0, 1)])) == xLength * (
                   yLength - 1)

    ## When yLength is odd, there is a possible tiling
    xLength = 3
    yLength = 2
    conDict = turn_into_constraintDict(get_all_sets(3, 2, {"vDomino": [(0, 0), (0, 1)]}))
    assert engine.contract(coreDict=get_constraintNetwork(conDict), openColors=[])[:] == (yLength % 2 == 0)