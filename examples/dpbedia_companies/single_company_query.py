from examples.dpbedia_companies import query_strings as qs
from tnreason.representation import workload_to_sparqlwrapper as wts

from tnreason.representation import suffixes as suf
from tnreason import engine

dbpediaEndpointString = "https://dbpedia.org/sparql"

polCoresDict, intDict = wts.queries_to_cores(dbpediaEndpointString,
                                                       {"com": qs.comp_branch_location_query},
                                             coreType="PolynomialCore")

print(polCoresDict["com"].shape)

branchLocationList = engine.contract(polCoresDict, openColors=["branch" + suf.terVarSuf,
                                                               "location" + suf.terVarSuf],
                                     contractionMethod="CorewiseContractor")
branchLocationList.add_identical_slices()
print(branchLocationList.values)

print(intDict)