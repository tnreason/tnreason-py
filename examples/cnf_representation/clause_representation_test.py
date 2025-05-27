from tnreason.representation import cnf_to_cores as ctc

for coreType in ["PolynomialCore", "NumpyCore", "PandasCore"]:
    clauseCore = ctc.clause_to_core({"c": 0, "b": 1}, coreType=coreType)
    print(clauseCore[0, 0] == 1, clauseCore[1, 0] == 0)
