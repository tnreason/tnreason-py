<h1 style="font-weight: normal;">
  <span style="font-size: 2.2rem; font-weight: bold;">tnreason:</span> 
  <span style="font-size: 2.2rem; font-weight: bold;">T</span>ensor 
  <span style="font-size: 2.2rem; font-weight: bold;">N</span>etworks for efficient and explainable 
  <span style="font-size: 2.2rem; font-weight: bold;">Reason</span>ing
</h1>

<table>
<tr>
<td style="width: 60%;">


`tnreason` is the implementation of the tensor network approach to efficient and explainable AI, documented in the [report](https://github.com/EnexaProject/enexa-tensor-reasoning-documentation/blob/master/tnreason_report.pdf).

## Installation

The latest version of `tnreason` can be installed from the Python Package Index (PyPI) using pip:

```bash
pip install tnreason
```

</td> 
<td  style="width: 40%;" align="center"> 
<img src="images/icon.png" alt="Library Icon" width="120" /> 
</td>
</tr> 
</table>

## Application Demonstrations

- **Hybrid Logic and Probabilistic Reasoning:**
  The tensor network formalism generalizes logical and probabilistic reasoning.
  It therefore enables the combination of hard logical constraints in probabilistic models, which is a form of hybrid
  reasoning.
  For a demonstration on hard and soft accounting rules see the [Accounting Example](https://github.com/EnexaProject/enexa-tensor-reasoning/tree/version1/demonstrations/accounting).

- **Statistical models of Knowledge Graphs:**
  Tensor networks are furthermore useful in storing Knowledge Graphs, and more general of worlds in first-order logic.
  Based on the sample extraction formalism described in Chapter 11 of
  the [report](https://github.com/EnexaProject/enexa-tensor-reasoning-documentation/blob/master/tnreason_report.pdf)
  hybrid logic networks can be trained on data extracted from a knowledge base.
  For a demonstration of this method on the DPpedia Knowlege Graph
  see the [DBpedia Example](https://github.com/EnexaProject/enexa-tensor-reasoning/tree/version1/demonstrations/statistical_models_kg).

- **Solution of Constraint Satisfaction Problems (CSP):**
  Local constraints can be captured by boolean tensors and CSPs consider contractions of these boolean tensors.
  Efficient message passing algorithms can be exploited in the solution of these problems.
  A particular well-known example of a CSP is the game of Sudoku,
  see the [Sudoku Example](https://github.com/EnexaProject/enexa-tensor-reasoning/tree/version1/demonstrations/sudoku).

## Architecture

<p align="left">
  <a href="images/architecture.png">
    <img src="images/architecture.png" alt="Library Icon" width="240" />
  </a>
</p>

For references to the implemented concepts see Appendix A in
the [report](https://github.com/EnexaProject/enexa-tensor-reasoning-documentation/blob/master/tnreason_report.pdf).

## References

Tutorials can be found here in
the [colab demonstrations](https://drive.google.com/drive/folders/1JeHH8haomh3fhtXY1KQYvF6iKZRDRCqG?usp=share_link).

A mathematical report can be found at
the [documentation repository](https://github.com/EnexaProject/enexa-tensor-reasoning-documentation).

