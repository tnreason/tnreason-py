# tnreason for Statistical Models of Knowledge Graphs

We here demonstrate the usage of `tnreason` for the extraction of sample data from Knowledge Graphs and the training of
statistical models given these data.

## Background

Tensor networks provide a storage method for Knowledge Graphs, and more general of worlds in first-order logic.
Based on the sample extraction formalism described in Chapter 11 of the [report](https://github.com/EnexaProject/enexa-tensor-reasoning-documentation/blob/master/tnreason_report.pdf)
hybrid logic networks can be trained on data extracted from a knowledge base.

## Approach

We need a
- `importanceQuery` marking object tuples which form a sample
- `extractionQueries` extracting assignments to atomic variables for each sample

## Examples

We demonstrate this method on the DBpedia Knowledge Graph, which is a large-scale knowledge base extracted from Wikipedia.
To this end we extract data from the [SPARQL endpoint](https://dbpedia.org/sparql) of DBpedia.


