from docplex.mp.model import Model


def core_to_cplex_model(core):
    """
    Need binary variables, i.e. leg dimension = 2!, Otherwise: Do atomization first.
    Docplex format especially useful for translation to qiskit
    """
    model = Model(str(core.name) + "_cplexModel")

    variableDict = {
        color: model.binary_var(name=color) for color in core.colors
    }

    slackVariableDict = {

    }

    j=0
    for entry in iter(core):
        slackVariableDict["slack" + str(j)] = model.binary_var(name="slack" + str(j))
        lowBound = 1
        for var in entry[1]:
            if entry[1][var] == 1:
                lowBound = lowBound + variableDict[var] - 1
                model.add_constraint(slackVariableDict["slack" + str(j)] <= variableDict[var])
            elif entry[1][var] == 0:
                lowBound = lowBound - variableDict[var]
                model.add_constraint(slackVariableDict["slack" + str(j)] <= (1 - variableDict[var]))
            else:
                raise ValueError("Index {} not supported, binary only!".format(entry[1][var]))
        model.add_constraint(lowBound <= slackVariableDict["slack" + str(j)])
        j+=1

    objective = 0
    for j, entry in enumerate(polyCore.values):
        objective = objective + entry[0] * slackVariableDict["slack" + str(j)]

    model.maximize(objective)
    return model


if __name__ == "__main__":
    from tnreason.representation import cnf_to_cores as ctc
    polyCore = ctc.weightedFormulas_to_sparseCore({
        "w1": ["imp", "a", "b", 0.678],
        "w2": ["a", 0.34]
    })
    print(polyCore)

    model = core_to_cplex_model(polyCore)

    exit()
    solution = model.solve()  ## Needs for usage IBM ILOG CPLEX Optimization Studio !

    # Print the solution
    if solution:
        print(f"Objective value: {solution.objective_value}")
        for color in polyCore.colors:
            print(f"x1 = {solution[color]}")
    else:
        print("No solution found")
