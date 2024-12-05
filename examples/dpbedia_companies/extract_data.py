from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd

from examples.dpbedia_companies import query_strings as qs
from tnreason.encoding import sparql_to_cores as stc
from tnreason import engine
dbpediaEndpointString = "https://dbpedia.org/sparql"


polCoresDict, intDict = stc.queries_to_polynomialCores(dbpediaEndpointString,
                                                       {"com": qs.companies_query,
                                                        "branche": qs.branches_query,
                                                        "size" : qs.size_query
                                                        })
engine.draw_factor_graph(polCoresDict)
print(polCoresDict["com_qCore"].shape)

# companiesCore, intDict = stc.query_to_polynomialCore(dbpediaEndpointString, qs.companies_query, startInterpretationDict=dict())
# print(intDict["company"][:5])
# branchesCore, intDict = stc.query_to_polynomialCore(dbpediaEndpointString, qs.branches_query, startInterpretationDict=intDict)
# print(intDict["company"][:5])
#
#
#
# exit()
# companies_data = stc.wrapper_json_evaluate_query(dbpediaEndpointString, qs.companies_query)
# branches_data = stc.wrapper_json_evaluate_query(dbpediaEndpointString, qs.branches_query)
# size_data = stc.wrapper_json_evaluate_query(dbpediaEndpointString, qs.size_query)
#
# companies = pd.json_normalize(companies_data['results']['bindings'])
# branches = pd.json_normalize(branches_data['results']['bindings'])
# sizes = pd.json_normalize(size_data['results']['bindings'])
#
#
#
# print(branches)
#
# print(branches["company.value"])
# # Query 3: Extract size information
#
#
# # Merge results into a single DataFrame
# merged_data = companies.merge(branches, on='company.value', how='left')
# merged_data = merged_data.merge(sizes, on='company.value', how='left')

# Analyze correlation (example: branch vs. employees)