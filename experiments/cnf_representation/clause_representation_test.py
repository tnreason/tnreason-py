from tnreason.encoding import cnf_to_cores as ctc

testClause = [{"c": 0, "b": 1}, {"a": 0}, {"d": 0, "a": 1}]

pcore1 = ctc.clause_to_core({"c": 0, "b": 1})
pcore2 = ctc.clause_to_core({"c": 0, "d": 1})

print(pcore2.values)

print(pcore2.shape, pcore1.shape)
print(pcore2.contract_with(pcore1).shape)
print(ctc.clauseList_to_core(testClause).shape)
