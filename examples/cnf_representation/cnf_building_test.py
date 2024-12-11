if __name__ == "__main__":
    from tnreason.encoding import cnf_to_cores as ctc

    testFormula = ["or", ["and", "b", "c"], ["and", "b", "c"]]
    cnf = ctc.to_cnf(testFormula, uppushAnd=False)
    print(cnf)
    cnf = ctc.to_cnf(testFormula, uppushAnd=True)
    print(cnf)

    testFormula = ["not", ["eq", "a", "b"]]
    cnf = ctc.to_cnf(testFormula)
    print(cnf)

    testFormula = ["eq", ["eq", "a", "b"], ["not", ["imp", "b", "c"]]]
    cnf = ctc.to_cnf(testFormula)
    print(cnf)
