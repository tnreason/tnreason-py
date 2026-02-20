"""
Horn clauses contain at most one positive literal.
The inference algorithm "forward chaining" is complete for Horn clauses.
Here we represent forward chaining as an efficient message passing framework, where inference clusters are assigned to single clauses.
We test the algorithm on an simple knowledge base, which has a unique model, and show that the algorithm finds this model.
"""

exampleClauseList=[
    {'1': 0, '2': 0, '3': 1},
    {'1': 0, '4': 1},
    {'2': 0, '4': 1},
    {'3': 0, '4': 1},
    {'4': 0, '5': 1},
    {'1': 1},
    {'2': 1},
    {'6': 0},
    {'7': 0, '6': 1}
]

knownAtStart = ["atomFeature_1","atomFeature_2","atomFeature_6"] # Those in clauses of length 1

from demonstrations.sat import sat_forward_mp as satmp

mpInferer = satmp.get_forward_mp_inferer(exampleClauseList)

mpInferer.propagate_until_convergence(nonTrivialFeatureKeys=knownAtStart, verbose=True)

assignment = satmp.retrieve_atoms(mpInferer)

assert "atomFeature_1" in assignment
assert "atomFeature_2" in assignment
assert "atomFeature_3" in assignment
assert "atomFeature_4" in assignment
assert "atomFeature_5" in assignment
assert "atomFeature_6" in assignment
assert "atomFeature_7" in assignment

assert assignment["atomFeature_1"]
assert assignment["atomFeature_2"]
assert assignment["atomFeature_3"]
assert assignment["atomFeature_4"]
assert assignment["atomFeature_5"]
assert not assignment["atomFeature_6"]
assert not assignment["atomFeature_7"]