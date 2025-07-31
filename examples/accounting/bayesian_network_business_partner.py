import pandas as pd


from tnreason.engine import workload_to_pandas as wtp
from tnreason import engine

def get_result_df(normedCoreValues, basisCoreValues, termCore):
    dfs_dict = []
    for i in range(len(normedCoreValues)):
        result_dict = {}
        for key, value in normedCoreValues[i][1].items():
            original_key = key.split("_tV")[0]
            original_value = termCore.interpretationDict[original_key][value]
            # print(key, value)
            # print(original_key, original_value)
            result_dict[original_key] = original_value
        result_dict['count'] = int(basisCoreValues[i][0])
        result_dict["probability %"] = round(normedCoreValues[i][0] * 100)

        dfs_dict.append(result_dict)

    return pd.DataFrame(dfs_dict)


path = "assets/toy_accounting_businessPartner.csv"
df = pd.read_csv(path)
print(df)


labels = [
 'ACCOUNT_tV',
 'BP_ACCOUNT_tV',
]
features = ['BUSINESSPARTNER_tV', 'VAT_tV']

termCore = wtp.PandasTermCore(valueDf=df)

pdCore = engine.convert(termCore, "PolynomialCore")

open_colors=features+labels
smallCore = engine.contract(
    coreDict={
        "pdCore":pdCore.clone()
    },
    openColors=open_colors,
   contractionMethod="CorewiseContractor"
        )

smallCore.add_identical_slices()


normedCore = engine.normalize({"core":smallCore.clone()}, outColors= labels,
                   inColors = features,
                   contractionMethod="CorewiseContractor",
                   coreType="PolynomialCore")


resultDf = get_result_df(normedCore.values, smallCore.values, termCore)



resultDf = resultDf.sort_values(["probability %", "count"], ascending=False)

print(resultDf)