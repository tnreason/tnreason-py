
def get_binary_selection_architecture(setSize, variableList):
    """
    Selecting a subset of maximal size setSize
    """
    positionNeurons = {
        "posNeur"+ str(k): [["not","id"], variableList]
        for k in range(setSize)
    }

    conjunctionNeurons = {
        "conNeur0" : [["and"], ["posNeur0"], ["posNeur1"]],
        **{"conNeur"+str(k): [["and"],["conNeur"+str(k-1)],["posNeur"+str(k+1)]] for k in range(1,setSize-1)}
    }

    return {**positionNeurons, **conjunctionNeurons}


def get_categorical_selection_architecture(variableList, variableDims):
    """
    Selecting a state to each variable
    """
    positionNeurons = {
        "posNeur"+ str(k): [["not","id"], variableList[k]+"=["+str(variableDims[k])+"]"]
        for k, variable in enumerate(variableList)
    }

    conjunctionNeurons = {
        "conNeur0" : [["and"], ["posNeur0"], ["posNeur1"]],
        **{"conNeur"+str(k): [["and"],["conNeur"+str(k-1)],["posNeur"+str(k+1)]] for k in range(1,len(variableList)-1)}
    }

    return {**positionNeurons, **conjunctionNeurons}

if __name__ == "__main__":
    from tnreason import representation, engine

    colorNum = 10
    selNum = 3
    binSelArchitecture = get_binary_selection_architecture(selNum, ["a" + str(i) for i in range(colorNum)])
    engine.draw_factor_graph(representation.create_architecture(binSelArchitecture, ["conNeur" + str(selNum - 2)]))

    catSelArchitecture = get_categorical_selection_architecture(["c","a","b"],[10, 20, 30])
    engine.draw_factor_graph(representation.create_architecture(catSelArchitecture, ["conNeur1"]))
