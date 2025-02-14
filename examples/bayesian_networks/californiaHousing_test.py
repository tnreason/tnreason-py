from tnreason.engine import workload_to_pandas as wtp
from tnreason import engine

import pandas as pd

from examples.bayesian_networks import parameter_estimation as pe

termCore = wtp.PandasTermCore(valueDf=pd.read_csv(
        "/Users/alexgoessmann/Documents/ENEXA/tnreason/version1/tests/assets/california_housing_test.csv"))

dataCores = {"dataCore": engine.convert(termCore, "PandasCore")}

features = ["households_tVar", 'longitude_tVar', 'latitude_tVar', 'housing_median_age_tVar',
                       'total_rooms_tVar']
estimated = pe.estimate_bn(dataCores, pe.get_naive_parentsDict(
        featureColors=features,
        classColor="median_house_value_tVar"),
                            contractionMethod="CorewiseContractor", coreType="PandasCore")

for feature in features:
    estimated[feature + "_estCore"].values.to_csv("./results/households_"+feature+".csv")

engine.draw_factor_graph(estimated)