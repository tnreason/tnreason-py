import kickuchi_contractor as kic

from tnreason import application, representation

samples = application.InferenceProvider(
    application.HybridKnowledgeBase(
        weightedFormulas={"w": ["imp", "a", "b", 1]}
    )
).draw_samples(10, dfOutput=True)

print(samples)
dataCores = representation.create_data_cores(samples, ["a", "b"])
print(dataCores)

kiki = kic.KikuchiContractor(
    colorDict={"data": ["j", "a"],
               "datb": ["j", "b"],
               "f": ["j", "a", "b", "(imp_a_b)"],
               "formula": ["(imp_a_b)"]},
    coreDict={"data": {"da" : dataCores["a_dataCore"]}, "datb" : {"db": dataCores["b_dataCore"]},
              "f" : {**representation.create_formulas_cores({"w": ["imp", "a", "b"]}), **representation.create_formula_head(["imp", "a", "b"], headType="truthEvaluation")},
              "formula": {}}
)

kiki.update_message("f", "datb")
kiki.update_message("f", "data")
kiki.update_message("f","formula")

print(kiki.messages["f"]["formula"]["f_formula_mesCore"].values)