import gurobipy as gp
from gurobipy import GRB


def optimize_gurobi_model(gurobiModel):
    gurobiModel.optimize()
    return {v.varName: v.x for v in gurobiModel.getVars()}


def core_to_gurobi_model(core):
    """
    Works on any iterateable TensorCore
    Need binary variables, i.e. leg dimension = 2!, Otherwise: Do atomization first.
    """

    model = gp.Model(str(core.name) + "_gurobiModel")

    variableDict = {
        color: model.addVar(vtype=GRB.BINARY, name=color) for color in core.colors
    }

    slackVariableDict = {}
    j = 0
    for entry in iter(core):
        slackVariableDict["slack" + str(j)] = model.addVar(vtype=GRB.BINARY, name="slack" + str(j))
        lowBound = 1
        for var in entry[1]:
            if entry[1][var] == 1:
                lowBound = lowBound + variableDict[var] - 1
                model.addConstr(slackVariableDict["slack" + str(j)] <= variableDict[var])
            elif entry[1][var] == 0:
                lowBound = lowBound - variableDict[var]
                model.addConstr(slackVariableDict["slack" + str(j)] <= (1 - variableDict[var]))
            else:
                raise ValueError("Index {} not supported, binary only!".format(entry[1][var]))
        model.addConstr(lowBound <= slackVariableDict["slack" + str(j)])
        j += 1

    objective = 0
    j = 0
    for entry in iter(core):
        objective = objective + entry[0] * slackVariableDict["slack" + str(j)]
        j += 1
    model.setObjective(objective, GRB.MAXIMIZE)
    return model
