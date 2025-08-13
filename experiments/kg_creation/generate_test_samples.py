from tnreason import application

dist = application.HybridKnowledgeBase(weightedFormulas={
    "f1": ["or", "alpha", "beta", 0.4],
    "f2": ["gamma", 0.3]
})
samples = application.InferenceProvider(dist).draw_samples(10, dfOutput=True)
samples["sample_index"] = samples.index
samples.to_csv("/Users/alexgoessmann/Documents/ENEXA/tnreason/version1/demonstrations/kg_creation/example.csv")

