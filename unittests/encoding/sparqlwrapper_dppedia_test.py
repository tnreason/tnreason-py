import unittest

from tnreason.encoding import sparql_to_cores as stc

class SparqlwrapperTest(unittest.TestCase):
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