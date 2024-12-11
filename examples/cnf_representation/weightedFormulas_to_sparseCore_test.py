from tnreason.encoding import cnf_to_cores as ctc

if __name__ == "__main__":

    print(ctc.weightedFormulas_to_sparseCore({
            "w1": ["imp", "a", "b", 2],
            "w2": ["or", "c", "d", 1.1]
        }))


    testFormula = ["xor", ["eq", "a", "b"], ["not", ["imp", "b", "c"]]]
    print(ctc.formula_to_sparseCore(testFormula))

    testFormula = ["eq", ["eq", "a", "b"], ["not", ["imp", "b", "c"]]]
    ctc.formula_to_sparseCore(testFormula)

    testFormula = ["or", ["and", "b", "c"], ["and", "b", "c"]]
    clauseList = ctc.cnf_to_dict(ctc.to_cnf(testFormula, uppushAnd=False))
    assert len(ctc.simplify_clauseList(clauseList)) == 2  # Needs to be {"b": 1}, {"c": 1}

    ctc.formula_to_sparseCore(testFormula)

    print(
        ctc.weightedFormulas_to_sparseCore({
            "w1": ["imp", "a", "b", 2],
            "w2": ["or", "c", "d", 1.1]
        })
    )
