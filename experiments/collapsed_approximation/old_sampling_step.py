from experiments.collapsed_approximation import old_selection_architecture_creation as sela

from tnreason import engine, representation

tbApproximated = representation.create_formulas_cores({"w1": ["imp", "a", "b"]})

neuronNameDict = sela.get_binary_selection_architecture(setSize=2, variableList=["a", "b"])

neuronColorDict = representation.parse_neuronNameDict_to_neuronColorDict(neuronNameDict)



contracted = engine.contract(coreDict={**tbApproximated, **representation.create_architecture(neuronColorDict, headNeuronNames=["conNeur0"])},
                             openColors=representation.find_selection_colors(neuronNameDict))
solution = contracted.get_argmax()

print(solution)
print(representation.create_solution_expression(neuronNameDict, solution))
print(representation.create_solution_expression(neuronColorDict, solution))
