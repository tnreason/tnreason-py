from tnreason.encoding import workload_to_sparqlwrapper as wts
from tnreason import engine

from examples.bayesian_networks import parameter_estimation as pe

dbpediaEndpointString = "https://dbpedia.org/sparql"
coreType = "PandasCore"

personQuery = """
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dct: <http://purl.org/dc/terms/>

SELECT ?person ?country ?language ?birthPlace ?nationality
WHERE {
        ?person rdf:type dbo:Person ;                         # Type of the entity
          dbo:country ?country ;                   # Country associated
          dbo:language ?language ;                 # Language information
          dbo:birthPlace ?birthPlace ;
          dbo:nationality ?nationality . 
}
LIMIT 1000
"""

qCoresDict, intDict = wts.queries_to_cores(dbpediaEndpointString, {"com": personQuery}, coreType=coreType)

parentsDict = {
    "nationality_tVar": ["birthPlace_tVar"],
    "language_tVar": ["nationality_tVar"],
    "birthPlace_tVar": []
}

bnCores = pe.estimate_bn(qCoresDict, parentsDict, contractionMethod="CorewiseContractor", coreType=coreType)

for coreKey in bnCores:
    print("##" + coreKey)

    # Only for PandasCores
    print(bnCores[coreKey].values.sort_values("values",ascending=False))

    # Only for PolynomialCores
    #for entry in bnCores[coreKey].values:
    #    print(entry)
