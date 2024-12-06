from examples.dpbedia_companies import query_strings as qs
from tnreason.encoding import sparql_to_cores as stc
from tnreason.encoding import categoricals_to_cores as ctc
from tnreason.encoding import suffixes as suf
from tnreason import engine

dbpediaEndpointString = "https://dbpedia.org/sparql"

polCoresDict, intDict = stc.queries_to_polynomialCores(dbpediaEndpointString,
                                                       {"com": qs.comp_branch_location_query})

print(polCoresDict["com_qCore"].shape)

branchLocationList = engine.contract(polCoresDict, openColors=["branch" + suf.termVariableSuffix,
                                                               "location" + suf.termVariableSuffix],
                                     method="PolynomialContractor")
branchLocationList.add_identical_slices()
print(branchLocationList.values)

print(intDict
