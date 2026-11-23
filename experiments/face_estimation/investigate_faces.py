import numpy as np
from scipy.spatial import ConvexHull


def get_facets(vertices):
    """
    Given a set of vertices, compute the convex hull and return the facets (faces).

    Parameters:
    vertices (np.ndarray): An array of shape (n, d) where n is the number of vertices and d is the dimension.

    Returns:
    list: A list of facets, where each facet is represented by a list of vertex indices.
    """
    hull = ConvexHull(vertices)
    return hull.simplices


def check_cube_like(all_vertices, face_vertices):
    cubeface_dict = propose_cube_face(face_vertices)
    intersection_vertices = [vertex for vertex in all_vertices if
                             all([vertex[i] == cubeface_dict[i] for i in cubeface_dict])]
    return len(intersection_vertices) == len(face_vertices)


def propose_cube_face(face_vertices):
    dim = len(face_vertices[0])
    cubeface_dict = {i: [] for i in range(dim)}

    for vertex in face_vertices:
        for i, coordinate in enumerate(vertex):
            if coordinate not in cubeface_dict[i]:
                cubeface_dict[i].append(coordinate)

    return {i: cubeface_dict[i][0] for i in cubeface_dict if len(cubeface_dict[i]) == 1}


if __name__ == "__main__":
    # 1. Define your vertices (x, y, z)
    vertices = np.array([
        [0, 0],  # 0
        [1, 0],  # 1
        [0, 1]  # 2
    ])
    facets = get_facets(vertices)
    assert len(facets) == 3

    assert check_cube_like(vertices, [[0, 0], [1, 0]])
    assert not check_cube_like(vertices, [[0, 1], [1, 0]])
