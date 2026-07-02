from tnreason import engine
from tnreason.application import formulas_to_cores as ftc


def extract_vertices_from_expressionsDict(expressionsDict):
    comCores = ftc.create_computation_cores_to_expressionDict(expressionsDict)
    headColors = [ftc.get_formula_headColor(expressionsDict[expKey]) for expKey in expressionsDict]

    contracted = engine.contract(comCores, openColors=headColors)

    return [[int(slice[1][key]) for key in slice[1]] for slice in contracted if slice[0] != 0], contracted.colors


## Naive by summing over the one-hot encoded vertices
def get_face_activation(vertices, headColors, shape=None, coreType="NumpyCore"):
    if shape is None:
        shape = [2 for _ in headColors]
    return engine.create_from_slice_iterator(shape=shape,
                                             colors=headColors,
                                             sliceIterator=[
                                                 (1, {headColor: vertex[i] for i, headColor in enumerate(headColors)})
                                                 for vertex in vertices],
                                             coreType=coreType)


def get_cubeface_activation(cubeDict, headColors, shape=None, coreType="NumpyCore"):
    """
    :param cubeDict: Specifying the \hardlegset by the keys and \headindexof{\hardlegset} by values
    """
    if shape is None:
        shape = [2 for _ in headColors]
    return engine.create_from_slice_iterator(shape=shape,
                                             colors=headColors,
                                             sliceIterator=[
                                                 (1, cubeDict)],
                                             coreType=coreType)


def test_entailment(mnCores, expressionsDict, faceVertices, headColors):
    partFunction = engine.contract(mnCores, openColors=[])[:]
    comValue = engine.contract({**mnCores, **ftc.create_computation_cores_to_expressionDict(expressionsDict),
                                "hardActCore": get_face_activation(faceVertices,headColors)
                                }, openColors=[])[:]
    return partFunction == comValue


if __name__ == "__main__":
    expressionsDict = {
        "f1": ["imp", "a", "b"],
        "f2": ["imp", "a", "c"],
        "f3": ["a"]}
    vertices, headColors = extract_vertices_from_expressionsDict(expressionsDict)

    assert test_entailment(
        {"ones": engine.create_from_slice_iterator([2, 2, 2], ["a_dV", "b_dV", "c_dV"], sliceIterator=[(1, dict())])}
        , expressionsDict, vertices, headColors)
    assert not test_entailment(
        {"ones": engine.create_from_slice_iterator([2, 2, 2], ["a_dV", "b_dV", "c_dV"], sliceIterator=[(1, dict())])}
        , expressionsDict, vertices[:3], headColors)

    assert test_entailment(
        {"aNever": engine.create_from_slice_iterator([2, 2, 2], ["a_dV", "b_dV", "c_dV"], sliceIterator=[(1, {"a_dV": 0})])}
        , expressionsDict, vertices, headColors)

    assert len(vertices) == 5
    assert len(headColors) == 3
    assert get_face_activation(vertices, headColors)[0, 1, 0] == 1
    assert get_cubeface_activation({"a": 1, "b": 0, "c": 1}, ["a", "b", "c", "d"])[1, 0, 1, 0] == 1
    assert get_cubeface_activation({"a": 1, "b": 0, "c": 1}, ["a", "b", "c", "d"])[1, 0, 1, 1] == 1
    assert get_cubeface_activation({"a": 1, "b": 0, "c": 1}, ["a", "b", "c", "d"])[1, 0, 0, 0] == 0

