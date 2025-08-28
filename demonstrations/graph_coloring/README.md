# tnreason for Graph Coloring

We here demonstrate the usage of `tnreason` for solving graph coloring problems using a tensor network approach.

## Background

Graph Coloring is an instance of a Constraint Satisfaction Problem, where the constraints are that neighboring nodes in a graph must be assigned different colors.
When the number of allowed colors exceeds 2, the problem is NP-complete.

## Approach

In `constraint_representation.py` we represent instances on a generic graph as a Computation-Activation Network.
The problem is then understood as a forward inference problem on the Computation-Activation Network.

## Examples

We investigate specific instance of graph coloring problems with 2 allowed colors in `examples/bipartite_decision.py`. 
These cases are efficiently solvable by message-passing along the edges and decide whether a  given graph is bipartite.