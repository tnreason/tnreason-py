from examples.dbpedia_companies import query_strings as qs
from tnreason.representation import workload_to_sparqlwrapper as wts
from tnreason import engine, representation

dbpediaEndpointString = "https://dbpedia.org/sparql"

polCoresDict, intDict = wts.queries_to_cores(dbpediaEndpointString,
                                             {"com": qs.comp_branch_location_query}, coreType="PolynomialCore")

convertedQueryCoresDict = {
    key: engine.convert(inCore=polCoresDict[key], outCoreType="NumpyCore") for key in
    polCoresDict
}

colorArchitecture = {
    "head": [["imp"], "location_tVar=[3]", "branch_tVar=[3]"]
}
selectionCoresDict = representation.create_architecture(colorArchitecture, headNeuronNames=["head"])

smallContracted = engine.contract({**selectionCoresDict, **convertedQueryCoresDict},
                                  openColors=["head_p0_vsVar",
                                              "head_p1_vsVar"])  # representation.find_selection_colors(colorArchitecture))
bestSelection = smallContracted.get_argmax()
print(smallContracted.values)
print(smallContracted.get_argmax())

fullContracted = engine.contract({**selectionCoresDict, **convertedQueryCoresDict},
                                 openColors=representation.find_selection_colors(colorArchitecture)).get_argmax()
print(representation.create_solution_expression(colorArchitecture, fullContracted))
