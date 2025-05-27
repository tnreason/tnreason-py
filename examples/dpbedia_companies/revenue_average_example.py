from examples.dpbedia_companies import query_strings as qs
from tnreason.representation import workload_to_sparqlwrapper as wts
from tnreason.representation import suffixes as suf
from tnreason import engine


dbpediaEndpointString = "https://dbpedia.org/sparql"

polCoresDict, intDict = wts.queries_to_cores(dbpediaEndpointString,
                                                       {"branch": qs.branches_query,
                                                        "size": qs.size_query
                                                        }, coreType="PolynomialCore")

import numpy as np
revenue_vector = engine.get_core("PolynomialCore")(values=np.array([float(entry) for entry in intDict["revenue"]]),
                                                   colors=["revenue" + suf.terVarSuf], name="RevenueValue")
print(revenue_vector.values)

branch_revenue_contraction = engine.contract({**polCoresDict, "RevenueValue": revenue_vector}, openColors=["branch" + suf.terVarSuf],
                                             contractionMethod="CorewiseContractor")
branch_revenue_contraction.add_identical_slices()
print(branch_revenue_contraction.values)