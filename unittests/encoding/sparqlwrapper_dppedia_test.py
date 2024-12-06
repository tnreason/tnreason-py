import unittest

from tnreason.encoding import sparql_to_cores as stc

class DBpediaTest(unittest.TestCase):
    def test_dppedia(self):

        results = stc.wrapper_json_evaluate_query(endpointString="https://dbpedia.org/sparql", queryString="""
            SELECT DISTINCT ?x ?y ?z
            WHERE { 
                ?x ?y ?z.
            }
            LIMIT 10
        """)

        entryPositionList, imDict = stc.query_result_to_entryPositionList(results, dict())
        self.assertTrue(len(entryPositionList) == 10)

    def test_categorical(self):
        results = stc.wrapper_json_evaluate_query(endpointString="https://dbpedia.org/sparql", queryString="""
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
        """)

        entryPositionList, imDict = stc.query_result_to_entryPositionList(results, dict())
