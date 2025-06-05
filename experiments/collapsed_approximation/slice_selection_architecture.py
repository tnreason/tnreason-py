import tnreason.representation.coordinate_calculus
from tnreason import representation, engine

from tnreason.representation import neurons_to_cores as ntc

"""
Differs from old binary_selection_architecture by the true passes u3 and a single land
"""


def get_and_core(headColorList):
    return engine.create_from_slice_iterator(shape=[2 for _ in headColorList], colors=headColorList,
                                             sliceIterator=[(1, {color: 1 for color in headColorList})])


def get_leg_neurons(variableColorList, order):
    architectureDict = {"posNeur" + str(k): [["not", "id", "u3"], variableColorList] for k in range(order)}
    return representation.create_architecture(architectureDict, headNeuronNames=[])


def create_slice_selecting_tn(variableColorList, sparsityOrder):
    return {**get_leg_neurons(variableColorList, order=sparsityOrder),
        "headAnd": get_and_core(
            ["posNeur" + str(k) + ntc.heaPre + representation.suf.comVarSuf for k in range(sparsityOrder)])}


def get_selection_colors(sparsityOrder):
    return ["posNeur" + str(k) + ntc.funPre for k in range(sparsityOrder)] + [
        "posNeur" + str(k) + ntc.posPre + "0" + representation.suf.selVarSuf for k in
        range(sparsityOrder)]

def get_selection_dimDict(sparsityOrder, variableNumber):
    return {**{"posNeur" + str(k) + ntc.funPre : 3 for k in range(sparsityOrder)}, # Missing suf.selVarSuf?
            **{"posNeur" + str(k) + ntc.posPre + "0" + representation.suf.selVarSuf: variableNumber for k in
               range(sparsityOrder)}
            }


if __name__ == "__main__":
    sparsityOrder = 2
    varNum = 3
    variableList = ["a" + str(i) for i in range(varNum)]

    selectionTN = create_slice_selecting_tn(variableList, sparsityOrder)
    contracted = engine.contract(selectionTN, openColors=["a0", "a1","a2"],
                                 colorEvidenceDict={selColor : 0 for selColor in get_selection_colors(sparsityOrder)},
                                 dimensionDict=get_selection_dimDict(sparsityOrder, varNum))

    assert contracted.values[0,1,0] == 1
    assert contracted.values[1,0,1] == 0

    exit()
    basePath = "/Users/alexgoessmann/Documents/ENEXA/tnreason/version1/experiments/collapsed_approximation/diagrams/"
    engine.draw_factor_graph(create_slice_selecting_tn(variableList, sparsityOrder),
                             storePath=basePath + "slice_selection_tn.png")
    engine.draw_factor_graph({**create_slice_selecting_tn(variableList, sparsityOrder),
                              "parCore" : tnreason.representation.coordinate_calculus.create_trivial_core(name="parCore",
                                                                                                          shape=[3 for _ in range(sparsityOrder)] + [2 for _ in range(sparsityOrder)],
                                                                                                          colors=get_selection_colors(sparsityOrder=2))},
                             storePath=basePath + "slice_energy_tn.png")