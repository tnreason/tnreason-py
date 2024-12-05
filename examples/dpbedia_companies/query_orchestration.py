from examples.dpbedia_companies import query_strings as qs
from tnreason.encoding import sparql_to_cores as stc
from tnreason.encoding import categoricals_to_cores as ctc
from tnreason.encoding import suffixes as suf
from tnreason import engine

dbpediaEndpointString = "https://dbpedia.org/sparql"

polCoresDict, intDict = stc.queries_to_polynomialCores(dbpediaEndpointString,
                                                       {"branch": qs.branches_query,
                                                        "size": qs.size_query
                                                        })

print(polCoresDict["branch" + suf.queryCoreSuffix].shape)
print(polCoresDict["size" + suf.queryCoreSuffix].shape)

branch_revenue_contraction = engine.contract(polCoresDict, openColors=["branch" + suf.termVariableSuffix,
                                                                       "revenue" + suf.termVariableSuffix],
                                             method="PolynomialContractor")
branch_revenue_contraction.add_identical_slices()

print(branch_revenue_contraction.shape)
print(branch_revenue_contraction.values)

## Atomization of the first branch
pos = 1
atomizationCoreDict = ctc.create_single_atomization(catColor="branch" + suf.termVariableSuffix,
                                                    catDim=len(intDict["branch"]), position=pos,
                                                    atomColor=intDict["branch"][pos])

engine.draw_factor_graph({**polCoresDict, **atomizationCoreDict})
