import unittest

from tnreason import reasoning, representation, application, engine
import pandas as pd
import numpy as np

from tnreason.application import script_transform as st

## Create toy data
sampleDf = pd.get_dummies(pd.DataFrame({'DOCUMENT_TYPE': ['Incoming invoice',
                                                          'Incoming invoice',
                                                          'Incoming invoice',
                                                          'Incoming invoice',
                                                          'Incoming invoice',
                                                          'Incoming invoice'],
                                        'PRODUCT_CLASS': ['SEA-CP_01',
                                                          'SEA-CP_02',
                                                          'SEA-CP_03',
                                                          'SEA-CP_04',
                                                          'SEA-CP_05',
                                                          'SEA-CP_01'],
                                        'TAX_RATE': ['19', '19', '19', '7', '19', '7'],
                                        'INDUSTRIAL_SECTOR': ['A', 'A', 'A', 'A', 'A', 'A'],
                                        'ACCOUNT': ['Account_3400',
                                                    'Account_3400',
                                                    'Account_3400',
                                                    'Account_3300',
                                                    'Account_4605',
                                                    'Account_3300']}))
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
                                                                       affectedComputationCores=list(
                                                                           catComCores.keys()),
                                                                       shape=[len(product_list), len(vat_list)]
                                                                       )},
    computationCoreDict=catComCores
)

## Calculate the satisfaction using forward inference
satisfactionDict = application.InferenceProvider(distribution=empDist.create_caNetwork()).ask_features(
    caNetwork.featureDict, computationCoreDict={**caNetwork.computationCoreDict, **caNetwork.baseMeasureCoreDict}
)

## Match the satisfaction using backward inference
binferer = reasoning.get_inferer("BackwardAlternator")(caNetwork=caNetwork,
                                                       meanParamDict=satisfactionDict)
binferer.alternating_updates(featureKeys=["productTaxHard"])

## Used in tests
currentModel = binferer.caNetwork
coarse_architecture = {
    "neur1": [["imp"],
              vat_list,
              account_list
              ]
}
fine_architecture = {
    "neur1": [["imp"],
              ["neur2"],
              account_list],
    "neur2": [["and"],
              ['TAX_RATE_19'],
              product_list]
}

class ToyAccountingTest(unittest.TestCase):
    def test_satisfactionDict_fInferer(self):
        self.assertTrue(satisfactionDict["productTaxHard"][0, 0] == 1)
        self.assertTrue(satisfactionDict["productTaxHard"][0, 1] == 1)
        self.assertTrue(satisfactionDict["productTaxHard"][1, 0] == 1)
        self.assertTrue(satisfactionDict["productTaxHard"][1, 1] == 0)
        self.assertTrue(satisfactionDict["productTaxHard"][2, 0] == 1)
        self.assertTrue(satisfactionDict["productTaxHard"][2, 1] == 0)
        self.assertTrue(satisfactionDict["productTaxHard"][3, 0] == 0)
        self.assertTrue(satisfactionDict["productTaxHard"][3, 1] == 1)
        self.assertTrue(satisfactionDict["productTaxHard"][4, 0] == 1)
        self.assertTrue(satisfactionDict["productTaxHard"][4, 1] == 0)

    def test_meanParamDict_bInferer(self):
        self.assertTrue(binferer.meanParamDict["productTaxHard"][0, 0] == 1)
        self.assertTrue(binferer.meanParamDict["productTaxHard"][0, 1] == 1)
        self.assertTrue(binferer.meanParamDict["productTaxHard"][1, 0] == 1)
        self.assertTrue(binferer.meanParamDict["productTaxHard"][1, 1] == 0)
        self.assertTrue(binferer.meanParamDict["productTaxHard"][2, 0] == 1)
        self.assertTrue(binferer.meanParamDict["productTaxHard"][2, 1] == 0)
        self.assertTrue(binferer.meanParamDict["productTaxHard"][3, 0] == 0)
        self.assertTrue(binferer.meanParamDict["productTaxHard"][3, 1] == 1)
        self.assertTrue(binferer.meanParamDict["productTaxHard"][4, 0] == 1)
        self.assertTrue(binferer.meanParamDict["productTaxHard"][4, 1] == 0)

    def test_boolean_architecture(self):
        """
        Check, whether the coarse and the fine architecture are boolean tensors
        """
        architectureDict = application.create_architecture(
            fine_architecture,
            headNeuronNames=["neur1"])
        contracted = engine.contract(architectureDict,
                                     openColors=application.find_selection_colors(
                                         fine_architecture) + st.add_color_suffixes(
                                         account_list) + st.add_color_suffixes(product_list) + [
                                                    st.add_color_suffixes(vat_list)[0]])
        for index in np.ndindex(*contracted.shape):
            self.assertTrue(contracted[{color: index[i] for i, color in enumerate(contracted.colors)}] in [0, 1])

        architectureDict = application.create_architecture(
            coarse_architecture,
            headNeuronNames=["neur1"])
        contracted = engine.contract(architectureDict,
                                     openColors=application.find_selection_colors(
                                         coarse_architecture) + st.add_color_suffixes(vat_list) + st.add_color_suffixes(
                                         account_list))
        for index in np.ndindex(*contracted.shape):
            self.assertTrue(contracted[{color: index[i] for i, color in enumerate(contracted.colors)}] in [0, 1])

    def test_likelihood_coarse(self):
        selVariables = application.find_selection_colors(coarse_architecture)
        partitionFunction = empDist.get_partition_function()
        self.assertTrue(partitionFunction == 6)
        positive_contracted = 1 / partitionFunction * engine.contract(coreDict={**empDist.create_cores(),
                                                                                **application.create_architecture(
                                                                                    coarse_architecture,
                                                                                    headNeuronNames=["neur1"])},
                                                                      openColors=selVariables)

        self.assertTrue(
            all([positive_contracted.shape[i] == [1, 2, 3][i] for i in range(len(positive_contracted.shape))]))
        for index in np.ndindex(*positive_contracted.shape):
            self.assertTrue(
                positive_contracted[{color: index[i] for i, color in enumerate(positive_contracted.colors)}] <= 1)
        self.assertAlmostEqual(positive_contracted[0, 0, 0], 0.33333, places=3)
        self.assertAlmostEqual(positive_contracted[0, 1, 2], 0.66666, places=3)

        currentPartitionFunction = currentModel.get_partition_function()
        self.assertTrue(currentPartitionFunction == 18)

        negative_contracted = 1 / currentPartitionFunction * engine.contract(
            coreDict={**currentModel.create_cores(),
                      **application.create_architecture(coarse_architecture, headNeuronNames=["neur1"])},
            openColors=selVariables)

        self.assertTrue(
            all([negative_contracted.shape[i] == [1, 2, 3][i] for i in range(len(negative_contracted.shape))]))
        for index in np.ndindex(*negative_contracted.shape):
            value = negative_contracted[{color: index[i] for i, color in enumerate(negative_contracted.colors)}]
            self.assertTrue(value <= 1 and value >= 0)

        self.assertAlmostEqual(negative_contracted[0, 0, 0], 0.555555, places=3)
        self.assertAlmostEqual(negative_contracted[0, 0, 2], 0.555555, places=3)
        self.assertAlmostEqual(negative_contracted[0, 1, 1], 0.777777, places=3)

        likelihood_gradient = positive_contracted + -1 * negative_contracted
        selMax = likelihood_gradient.get_argmax()  # Using that likelihood_gradient is a NumpyCore
        learnedFormula = application.create_solution_expression(coarse_architecture, selMax)["neur1"]

        self.assertEqual(learnedFormula, ["imp","TAX_RATE_19", "ACCOUNT_Account_3400"])