from tnreason.engine import workload_to_pandas as wtp
from tnreason import engine

import pandas as pd

from examples.bayesian_networks import parameter_estimation as pe

termCore = wtp.PandasTermCore(valueDf=pd.read_csv(
        "/Users/alexgoessmann/Documents/ENEXA/tnreason/version1/tests/assets/california_housing_test.csv").head())

dataCores = {"dataCore": engine.convert(termCore, "PandasCore")}

estimated = pe.estimate_bn(dataCores, pe.get_naive_parentsDict(
        featureColors=["households_tVar", 'longitude_tVar', 'latitude_tVar', 'housing_median_age_tVar',
                       'total_rooms_tVar'],
        classColor="median_house_value_tVar"),
                            contractionMethod="CorewiseContractor", coreType="PandasCore")

engine.draw_factor_graph(estimated)