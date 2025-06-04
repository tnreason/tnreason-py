from tnreason import reasoning, representation, application
import pandas as pd

from tnreason.application.inductive import calculate_satisfactionDict


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

empDist = application.get_empirical_distribution(sampleDf, atomColumns=vat_list + account_list)

hybridKB = application.load_kb_from_yaml("assets/coarse_model.yaml")

categoricalConstraints = {"account": account_list, "tax": vat_list, }
catComCores = representation.create_categorical_cores({
    catColor: representation.add_color_suffixes(categoricalConstraints[catColor]) for catColor in
    categoricalConstraints})

mnFeatures = {"accountTaxHard": reasoning.HardPartitionFeature(featureColors=["account", "tax"],
                                                               affectedComputationCores=list(catComCores.keys()),
                                                               interpretationDict={"account": account_list,
                                                                                   "tax": vat_list}),
              # "accountTaxSoft": reasoning.SoftPartitionFeature(featureColors=["account", "tax"],
              #                                                 affectedComputationCores=list(catComCores.keys()),
              #                                                 interpretationDict={"account": account_list,
              #                                                                     "tax": vat_list})
              }

satisfactionDict = {**calculate_satisfaction_on_computationLess(empDist,
                                                                featureDict=mnFeatures,
                                                                comCores=catComCores),
                    **calculate_satisfactionDict(empDist,
                                                 {**{expressionKey: hybridKB.weightedFormulas[
                                                                        expressionKey][:-1] for expressionKey in
                                                     hybridKB.weightedFormulas},
                                                  })}

caNet = hybridKB.create_caNetwork()
caNet.include_features(
    mnFeatures,
    computationCores=catComCores,
    canParamDict={"accountTaxHard": satisfactionDict["accountTaxHard"]}
)

inferer = reasoning.get_inferer("BackwardAlternator")(caNetwork=caNet,
                                                      meanParamDict=satisfactionDict
                                                      )
inferer.alternating_updates(featureKeys=["accountTaxHard", "coarse_1"])

print(inferer.meanParamDict)
