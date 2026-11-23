from experiments.face_estimation import investigate_faces as infa
from experiments.face_estimation import compute_polytope as copo

class FaceSearcher:
    def __init__(self, expressionsDict,  vertices=None):
        """
        :param vertices: list of vertices coordinates
        :param distributionCores: Markov network cores


        wlacz zmywarke
        """
        self.vertices, self.headColors = copo.extract_vertices_from_expressionsDict(expressionsDict)
        self.expressionsDict = expressionsDict

    def refine(self, tobeRefinedVertices, distributionCores):
        """
        Treat face as polytope, go to facets and check for entailment
        """
        facetVertexLists = infa.get_facets(tobeRefinedVertices)
        for facetVertexList in facetVertexLists:
            testVertices = [tobeRefinedVertices[i] for i in facetVertexList]
            if copo.test_entailment(distributionCores, self.expressionsDict, testVertices, self.headColors):
                print(testVertices)
                return True, testVertices
        return False, tobeRefinedVertices

    def find_face(self, distributionCores):
        refinedVertices = self.vertices
        refining = True
        while refining:
            refining, refinedVertices = self.refine(refinedVertices, distributionCores)
        return refinedVertices

if __name__ == "__main__":
    from tnreason import engine
    expressionsDict = {
        "f1": ["imp", "a", "b"],
        "f2": ["imp", "a", "c"],
        "f3": ["a"]}

    distributionCores = {"ones": engine.create_from_slice_iterator([2, 2, 2], ["a_dV", "b_dV", "c_dV"], sliceIterator=[(1, dict())])}
    fs = FaceSearcher(expressionsDict)
    print(fs.find_face(distributionCores))

    ## Scipy fails to find facets of smaller dimension!
    print(fs.find_face(
        {"aNever": engine.create_from_slice_iterator([2, 2, 2], ["a_dV", "b_dV", "c_dV"], sliceIterator=[(1, {"a_dV": 0})])}
    ))

