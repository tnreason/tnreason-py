from experiments.collapsed_approximation import old_selection_architecture_creation as sela

from tnreason import engine, encoding

tbApproximated = encoding.create_formulas_cores({"w1": ["imp", "a", "b"]})

neuronNameDict = sela.get_binary_selection_architecture(setSize=2, variableList=["a", "b"])

neuronColorDict = encoding.parse_neuronNameDict_to_neuronColorDict(neuronNameDict)



contracted = engine.contract(coreDict={**tbApproximated, **encoding.create_architecture(neuronColorDict, headNeuronNames=["conNeur0"])},
                             openColors=encoding.find_selection_colors(neuronNameDict))
solution = contracted.get_argmax()

print(solution)
print(encoding.create_solution_expression(neuronNameDict, solution))
print(encoding.create_solution_expression(neuronColorDict, solution))
