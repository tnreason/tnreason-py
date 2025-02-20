from tnreason import engine


def get_weightedSumFunction(varColorList, varPmDimDict, pmCutOff, weights=None):
    """
    weights: Used to control, whether variable summed or substracted (by special cases of +1/-1 entries)
    """
    if weights is None:
        weights = [1 for _ in varColorList]
    return lambda *x: [
        min(max(sum([weights[i] * (x[i] + varPmDimDict[color][0]) for i, color in enumerate(varColorList)]),
                pmCutOff[0]),
            pmCutOff[1]) - pmCutOff[0]]


def get_weightSelectingFunction(varColorList, varPmDimDict, pmCutOff, weightSuperList):
    return lambda *x: get_weightedSumFunction(varColorList, varPmDimDict, pmCutOff, weights=weightSuperList[x[-1]])(
        *x[:-1])


def create_sum(varColorList, varPmDimDict, headColor, pmCutOff):
    return engine.create_relational_encoding(
        inshape=[varPmDimDict[color][1] - varPmDimDict[color][0] for color in varColorList],
        outshape=[pmCutOff[1] - pmCutOff[0]],
        incolors=varColorList,
        outcolors=[headColor],
        function=get_weightedSumFunction(varColorList, varPmDimDict, pmCutOff))


def create_sum_selector(varColorList, varPmDimDict, headColor, pmCutOff, weightSuperList, selectionColor,
                        coreType="NumpyCore"):
    """
    Creates sums of selected weights
    """
    return engine.create_relational_encoding(
        inshape=[varPmDimDict[color][1] - varPmDimDict[color][0] for color in varColorList] + [len(weightSuperList)],
        outshape=[pmCutOff[1] - pmCutOff[0]],
        incolors=varColorList + [selectionColor],
        outcolors=[headColor],
        function=get_weightSelectingFunction(varColorList, varPmDimDict, pmCutOff, weightSuperList),
        coreType=coreType)


if __name__ == "__main__":
    varDimDict = {"a": [-5, 10], "b": [-5, 10]}
    varColorList = ["a", "b"]
    pmCutOff = [-10, 20]

    pFunc = get_weightedSumFunction(["a", "b"], varDimDict, [-10, 20])
    assert pFunc(15, 15) == [30]

    weightSuperList = [[-1, 1], [1, -1], [1, 1]]
    selFunc = get_weightSelectingFunction(["a", "b"], varDimDict, [-10, 20], weightSuperList)
    assert selFunc(15, 15, 2) == [30]

    selCore = create_sum_selector(["a", "b"], varDimDict, "a+-b", [-10, 20], weightSuperList, "a+-_sVar", "PolynomialCore")
    assert selCore.colors == ['a', 'b', 'a+-_sVar', 'a+-b']
    assert selCore.shape == [15, 15, 3, 30]

    print(len(selCore.values))
    sumCore = create_sum(["a", "b"], varDimDict, "a+b", [-10, 20])
