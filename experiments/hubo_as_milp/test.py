from tnreason import engine

polyCore = engine.get_core("PolynomialCore")(values=[(-2, {"c1": 0, "c2": 1}), (3.4, {"c2": 0})],
                                             shape=[3, 2],
                                             colors=["c1", "c2"])

from tnreason.engine import polynomial_handling as ph

binP = ph.binarize_polyCore(polyCore)
print(binP.colors)
print(binP.values)

print(polyCore.get_argmax())

if __name__ == "__main__":
    from experiments.cnf_representation import formula_to_polynomial_core as ftp

    polyCore = ftp.weightedFormulas_to_polynomialCore({
        "w1": ["imp", "a", "b", 0.678],
        "w2": ["a", 0.34]
    })
    print(polyCore)

    model = engine.poly_to_gurobi.poly_to_gurobi_model(polyCore)
    model.optimize()

    # Output the results
    for v in model.getVars():
        print(f'{v.varName}: {v.x}')
