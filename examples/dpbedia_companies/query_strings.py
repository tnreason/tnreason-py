companies_query = """
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?company ?label
WHERE {
  ?company a dbo:Company .
  ?company rdfs:label ?label .
  FILTER (LANG(?label) = "en")
}
LIMIT 10
"""

branches_query = """
PREFIX dbo: <http://dbpedia.org/ontology/>
SELECT DISTINCT ?company ?branch
WHERE {
  ?company a dbo:Company .
  OPTIONAL { ?company dbo:industry ?branch . }
  OPTIONAL { ?company dbo:field ?branch . }
}
LIMIT 10
"""

size_query = """
PREFIX dbo: <http://dbpedia.org/ontology/>
SELECT DISTINCT ?company ?employees ?revenue
WHERE {
  ?company a dbo:Company .
  OPTIONAL { ?company dbo:numberOfEmployees ?employees . }
  OPTIONAL { ?company dbo:revenue ?revenue . }
}
LIMIT 100
"""