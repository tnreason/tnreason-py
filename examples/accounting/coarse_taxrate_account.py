import pandas as pd
import numpy as np
from tnreason import application, representation, engine

sampleDf = pd.read_csv("assets/toy_accounting.csv")
vat_list = [col for col in sampleDf.columns if 'TAX_RATE' in col]
account_list = [col for col in sampleDf.columns if 'ACCOUNT' in col]

empDist = application.get_empirical_distribution(sampleDf, atomColors=vat_list + account_list)
currentModel = application.HybridKnowledgeBase(categoricalConstraints={"tax" : vat_list, "account": account_list})

## Define selection architecture and current model
coarse_architecture = {
    "neur1" : [["imp"],
               vat_list,
               account_list
    ]
               }
selVariables = representation.find_selection_colors(coarse_architecture)

## Calculate likelihood gradient for grafting
positive_phase = {**empDist.create_cores(),
                  **representation.create_architecture(coarse_architecture, headNeuronNames=["neur1"])}
positive_contracted = 1/empDist.get_partition_function() * engine.contract(coreDict=positive_phase, openColors=selVariables)

negative_phase = {**currentModel.create_cores(),
                  **representation.create_architecture(coarse_architecture, headNeuronNames=["neur1"])}
negative_contracted = 1/currentModel.get_partition_function() * engine.contract(coreDict=negative_phase, openColors=selVariables)

likelihood_gradient = positive_contracted + -1*negative_contracted

## Extract formulas in gradient by threshold criterion
learnedFormulas = dict()
threshold = 0.1
i = 0
while np.max(likelihood_gradient.values) > threshold:
    selMax = likelihood_gradient.get_argmax()
    learnedFormulas["coarse_"+str(i)] = representation.create_solution_expression(coarse_architecture, selMax)["neur1"]
    i+=1
    likelihood_gradient[selMax] = -1*likelihood_gradient[selMax]


## Build a model with these formulas, calibrate weights on data
currentModel.include(application.HybridKnowledgeBase(weightedFormulas={
    **{key: learnedFormulas[key]+[0] for key in learnedFormulas}
}))

weightEstimator = application.WeightEstimator(currentModel)
weightEstimator.get_satisfaction_dict(empDist)
weightEstimator.fact_check()
weightEstimator.calibrate_weights(10)

currentModel.to_yaml("assets/coarse_model.yaml")
print(currentModel)

