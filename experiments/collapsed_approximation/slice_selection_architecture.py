from tnreason import encoding, engine

"""
Differs from old binary_selection_architecture by the true passes u3 and a single land
"""


def get_and_core(headColorList):
    return engine.create_from_slice_iterator(shape=[2 for _ in headColorList], colors=headColorList,
                                             sliceIterator=[(1, {color: 1 for color in headColorList})])


def get_leg_neurons(variableColorList, order):
    architectureDict = {"posNeur" + str(k): [["not", "id", "u3"], variableColorList] for k in range(order)}
    return encoding.create_architecture(architectureDict, headNeuronNames=[])


def create_slice_selecting_tn(variableColorList, sparsityOrder):
    return {**get_leg_neurons(variableColorList, order=sparsityOrder),
            "headAnd": get_and_core(
                ["posNeur" + str(k) + encoding.suf.neurVariableSuffix for k in range(sparsityOrder)])}


if __name__ == "__main__":
    sparsityOrder = 2
    varNum = 1
    variableList = ["a" + str(i) for i in range(varNum)]
    engine.draw_factor_graph(create_slice_selecting_tn(variableList, sparsityOrder))
