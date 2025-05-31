import numpy as np
import pandas as pd
from tnreason import application, representation, engine

## Load data, now extended to the product atoms
sampleDf = pd.read_csv("assets/toy_accounting.csv")
vat_list = [col for col in sampleDf.columns if 'TAX_RATE' in col]
account_list = [col for col in sampleDf.columns if 'ACCOUNT' in col]
product_list = [col for col in sampleDf.columns if 'PRODUCT_CLASS' in col]

empDist = application.get_empirical_distribution(sampleDf, atomColors=vat_list + account_list + product_list)

## Load current model, extend it to product atoms
fineModel = application.load_kb_from_yaml("assets/coarse_model.yaml")
fineModel.include(application.HybridKnowledgeBase(categoricalConstraints={"product": product_list}))

fine_architecture = {
    "neur1": [["imp"],
              ["neur2"],
              account_list],
    "neur2": [["and"],
              ['TAX_RATE_19'],
              product_list]
}
selVariables = representation.find_selection_colors(fine_architecture)

## Calculate likelihood gradient for grafting
positive_phase = {**empDist.create_cores(),
                  **representation.create_architecture(fine_architecture, headNeuronNames=["neur1"])}
positive_contracted = 1 / empDist.get_partition_function() * engine.contract(coreDict=positive_phase,
                                                                             openColors=selVariables)

negative_phase = {**fineModel.create_cores(),
                  **representation.create_architecture(fine_architecture, headNeuronNames=["neur1"])}
negative_contracted = 1 / fineModel.get_partition_function() * engine.contract(
    coreDict=negative_phase, openColors=selVariables)

likelihood_gradient = positive_contracted + -1 * negative_contracted

## Extract formulas in gradient by threshold criterion
learnedFormulas = dict()
threshold = 0.08
i = 0
while np.max(likelihood_gradient.values) > threshold:
    selMax = likelihood_gradient.get_argmax()
    learnedFormulas["fine_"+str(i)] = representation.create_solution_expression(fine_architecture, selMax)["neur1"]
    i+=1
    likelihood_gradient[selMax] = -1*likelihood_gradient[selMax]

fineModel.include(application.HybridKnowledgeBase(weightedFormulas={
    **{key: learnedFormulas[key]+[0] for key in learnedFormulas}
}))


hybridLearner = application.HybridLearner(fineModel)
hybridLearner.infer_weights_on_data(empDist, satInferenceMethod="ForwardContractor")
hybridLearner.knowledgeBase.to_yaml("assets/fine_model.yaml")