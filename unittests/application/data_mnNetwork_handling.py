from tnreason import application
from tnreason import representation
from tnreason import engine

import pandas as pd

assetsBasePath = "/Users/alexgoessmann/Documents/ENEXA/tnreason/version1/unittests/knowledge/assets/"

backKb = application.load_kb_from_yaml(assetsBasePath + "fb_backKb.yaml")
architecture = application.load_from_yaml(assetsBasePath + "fb_architecture.yaml")

genKB = application.HybridKnowledgeBase(
    facts={"f1": ["a1"]},
    weightedFormulas={
        "wf1": ["imp", "a1", "a2", 1.1424],
        "wf1.5": ["a2", 0.2],
        "wf2": ["not", "a3", 5.2]
    }
)
sampleDf = application.InferenceProvider(genKB).draw_samples(100, dfOutput=True)

dataMN = application.get_empirical_distribution(sampleDf)
cores = dataMN.create_caNetwork(hardOnly=False).create_cores()

dist = application.get_empirical_distribution(pd.read_csv(assetsBasePath + "fb_sampleDf.csv"), atomColumns=["a1","a2","a3","a4"])

dist.create_caNetwork()
