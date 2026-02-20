from tnreason import application, representation, engine
import pandas as pd

def get_true_from_dict(dictionary):
    key_list = []
    for key, value in dictionary.items():
        if value:
            key_list.append(key)
    if len(key_list)==1:
        return key_list[0]
    elif len(key_list)==0:
        return None
    else:
        print(">1 True-Value found")
        return key_list


if __name__ == '__main__':
    Model = application.load_kb_from_yaml("assets/fine_model.yaml")

    y_true = ['Account_3400',
              'Account_3400',
              'Account_3400',
              'Account_3300',
              'Account_4605',
              'Account_3300']

    x_dict_list = [{'PRODUCT_CLASS_SEA-CP_01': True, 'TAX_RATE_19': True},
             {'PRODUCT_CLASS_SEA-CP_02': True, 'TAX_RATE_19': True},
             {'PRODUCT_CLASS_SEA-CP_03': True, 'TAX_RATE_19': True},
             {'PRODUCT_CLASS_SEA-CP_04': True, 'TAX_RATE_7': True},
             {'PRODUCT_CLASS_SEA-CP_05': True, 'TAX_RATE_19': True},
             {'PRODUCT_CLASS_SEA-CP_01': True, 'TAX_RATE_7': True}]

    sampleDf = pd.read_csv("assets/toy_accounting.csv")
    vat_list = [col for col in sampleDf.columns if 'TAX_RATE' in col]
    account_list = [col for col in sampleDf.columns if 'ACCOUNT' in col]

    account_list = ['ACCOUNT_Account_3300', 'ACCOUNT_Account_3400', 'ACCOUNT_Account_4605']

    result_dict_list = []
    for x in x_dict_list:
         result_dict_list.append(application.InferenceProvider(Model).exact_map_query(account_list, evidenceDict=x))

    print(result_dict_list)

    prediction_list = []
    for pred_dict in result_dict_list:
        predicted_label = get_true_from_dict(pred_dict)
        prediction_list.append(predicted_label)
    prediction_list = [pred.split("ACCOUNT_")[1] for pred in prediction_list]

    total_elements = len(y_true)
    matching_elements = sum(1 for x, y in zip(prediction_list, y_true) if x == y)

    # calculate accuracy
    accuracy = (matching_elements / total_elements) * 100
    print(f"The accuracy of the lists matching is: {round(accuracy, 2)}%")

    # get index of wrong prediction
    mismatched_indices = [i for i, (x, y) in enumerate(zip(prediction_list, y_true)) if x != y]

    if len(mismatched_indices) > 0:
        print("\nIndices of mismatched elements:")
        for index in mismatched_indices:
            print("index: ", index)
            print(f"predicted: {prediction_list[index]} - true: {y_true[index]}\n")
