comp_branch_location_query = """
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?company ?branch ?location
WHERE {
  ?company a dbo:Company .
  ?company dbo:industry ?branch .
  ?company dbo:location ?location
  FILTER (?branch IN (<http://dbpedia.org/resource/Bank>, 
                       <http://dbpedia.org/resource/Computer_Software>, 
                       <http://dbpedia.org/resource/Manufacturing>))
  FILTER (?location IN (<http://dbpedia.org/resource/United_States>, 
                       <http://dbpedia.org/resource/Germany>, 
                       <http://dbpedia.org/resource/Spain>))
}
"""

branches_query = """
PREFIX dbo: <http://dbpedia.org/ontology/>
SELECT DISTINCT ?company ?branch
WHERE {
  ?company a dbo:Company .
  ?company dbo:industry ?branch .
}
"""

size_query = """
PREFIX dbo: <http://dbpedia.org/ontology/>
SELECT DISTINCT ?company ?employees ?revenue
WHERE {
  ?company a dbo:Company .
  ?company dbo:numberOfEmployees ?employees . 
  ?company dbo:revenue ?revenue . 
}
LIMIT 100
"""