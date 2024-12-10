from tnreason.engine import workload_to_tentris as wt

termCore = wt.TentrisTripleStoreTermCore()
termCore.load_rdf_data("/home/version1/tentris_tests/bbb_generated.ttl")
termCore.adjust_interpretationsDict()

queryString = """
PREFIX dbo: <http://dbpedia.org/ontology/>
SELECT DISTINCT ?x ?y
WHERE {
  ?x ?z ?y.
}
LIMIT 5
"""

sparqlCore = termCore.eval_query(queryString, variables=["funx","randomy"])

from tnreason import engine
print(engine.convert(sparqlCore, "PolynomialCore").values)