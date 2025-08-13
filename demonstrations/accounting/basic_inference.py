from tnreason import reasoning, representation, application, engine
import pandas as pd
import numpy as np

from tnreason.application import script_transform as st

def calculate_satisfaction_on_computationLess(empDistribution, featureDict, comCores=dict(), inferenceMethod=None):
    empCaNet = empDistribution.create_caNetwork()

    empCaNet.include_features(
        featureDict=featureDict,
        computationCores=comCores
    )

    fInferer = reasoning.get_inferer(inferenceMethod)(caNetwork=empCaNet)
    fInferer.infer_meanParams(featureKeys=featureDict.keys())

    return {featureKey: fInferer.meanParamDict[featureKey] for featureKey in featureDict}

sampleDf = pd.read_csv("assets/toy_accounting.csv")
account_list = [col for col in sampleDf.columns if 'ACCOUNT' in col]
vat_list = [col for col in sampleDf.columns if 'TAX_RATE' in col]
product_list = [col for col in sampleDf.columns if 'PRODUCT' in col]
empDist = application.get_empirical_distribution(sampleDf, atomColumns=vat_list + account_list + product_list)

categoricalConstraints = {"account": account_list, "tax": vat_list, "product": product_list}
catComCores = application.create_categorical_cores({
    catColor: st.add_color_suffixes(categoricalConstraints[catColor]) for catColor in
    categoricalConstraints})

caNetwork = representation.ComputationActivationNetwork(
    featureDict={"productTaxHard": representation.HardPartitionFeature(featureColors=["product", "tax"],
                                                                       affectedComputationCores=list(catComCores.keys()),
                                                                       shape=[len(product_list), len(vat_list)]
                                                                       )},
    computationCoreDict=catComCores
)

inferer = reasoning.get_inferer("BackwardAlternator")(caNetwork=caNetwork,
                                                      meanParamDict=calculate_satisfaction_on_computationLess(empDist, caNetwork.featureDict, {**caNetwork.computationCoreDict,
                                                                                                              **caNetwork.baseMeasureCoreDict}))
inferer.alternating_updates(featureKeys=["productTaxHard"])

assert inferer.meanParamDict["productTaxHard"][0,0] == 1
assert inferer.meanParamDict["productTaxHard"][0,1] == 1
assert inferer.meanParamDict["productTaxHard"][1,0] == 1
assert inferer.meanParamDict["productTaxHard"][1,1] == 0
assert inferer.meanParamDict["productTaxHard"][2,0] == 1
assert inferer.meanParamDict["productTaxHard"][2,1] == 0
assert inferer.meanParamDict["productTaxHard"][3,0] == 0
assert inferer.meanParamDict["productTaxHard"][3,1] == 1
assert inferer.meanParamDict["productTaxHard"][4,0] == 1
assert inferer.meanParamDict["productTaxHard"][4,1] == 0

currentModel = inferer.caNetwork
fine_architecture = {
    "neur1": [["imp"],
              ["neur2"],
              account_list],
    "neur2": [["and"],
              ['TAX_RATE_19'],
              product_list]
}
selVariables = application.find_selection_colors(fine_architecture)

architectureCores = application.create_architecture(fine_architecture, headNeuronNames=["neur1"])


positive_phase = {**empDist.create_cores(),
                  **application.create_architecture(fine_architecture, headNeuronNames=["neur1"])}

partitionFunction = empDist.get_partition_function()
assert partitionFunction == 6
positive_contracted = 1/partitionFunction * engine.contract(coreDict={**empDist.create_cores(),
                  **application.create_architecture(fine_architecture, headNeuronNames=["neur1"])}, openColors=selVariables)

negative_phase = {**currentModel.create_cores(),
                  **application.create_architecture(fine_architecture, headNeuronNames=["neur1"])}
negative_contracted = 1/currentModel.get_partition_function() * engine.contract(coreDict=negative_phase, openColors=selVariables)

likelihood_gradient = positive_contracted + -1*negative_contracted

#selMax = likelihood_gradient.get_argmax()
#refinementFormula = application.create_solution_expression(fine_architecture, selMax)["neur1"]

## Extract formulas in gradient by threshold criterion
learnedFormulas = dict()
threshold = 0.1
i = 0
while np.max(likelihood_gradient.values) > threshold: ## To be integrated in HybridLearner: Use the grafting method
    selMax = likelihood_gradient.get_argmax()
    learnedFormulas["coarse_"+str(i)] = application.create_solution_expression(fine_architecture, selMax)["neur1"]
    i+=1
    likelihood_gradient[selMax] = -1*likelihood_gradient[selMax]
