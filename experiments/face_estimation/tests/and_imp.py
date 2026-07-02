from experiments.face_estimation import investigate_faces as infa
from experiments.face_estimation import compute_polytope as copo

expressionsDict = {
    "f1" : ["and", "a", "b"],
    "f2" : ["imp", "a", "b"],
}

vertices, headColors = copo.extract_vertices_from_expressionsDict(expressionsDict)
facets = infa.get_facets(vertices)

## 2 cube-like faces, 1 not cube-like face
assert sum([infa.check_cube_like(vertices, [vertices[i] for i in facet]) for facet in facets])==2