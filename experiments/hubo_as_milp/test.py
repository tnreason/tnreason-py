from tnreason import engine
from tnreason.engine import polynomial_handling as ph
from tnreason.engine import poly_to_gurobi as ptg
from experiments.cnf_representation import formula_to_polynomial_core as ftp

polyCore = engine.get_core("PolynomialCore")(values=[(-2, {"c1": 0, "c2": 1}), (3.4, {"c2": 0})],
                                             shape=[3, 2],
                                             colors=["c1", "c2"])

binP = ph.binarize_polyCore(polyCore)
model = ptg.core_to_gurobi_model(binP)
model.optimize()
# Output the results
for v in model.getVars():
    print(f'{v.varName}: {v.x}')


polyCore = ftp.weightedFormulas_to_polynomialCore({
        "w1": ["imp", "a", "b", 0.678],
        "w2": ["a", 0.34]
    })
print(polyCore)

model = ptg.core_to_gurobi_model(polyCore)
model.optimize()

# Output the results
for v in model.getVars():
    print(f'{v.varName}: {v.x}')
