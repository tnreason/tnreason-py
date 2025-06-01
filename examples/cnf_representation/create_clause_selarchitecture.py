import tnreason.representation.basis_calculus
from tnreason import representation, engine

"""
Provides a selection architecture to parametrize clauses
"""

def get_cnf_architecture(maxSize, atomList):
    return {
        **{"colNeur" + str(i): [["or"], ["neur" + str(i + 1)], ["colNeur" + str(i - 1)]] for i in
           range(1, maxSize - 1)},  # collection neuron, headNeuron is
        "colNeur0": [["or"], ["neur0"], ["neur1"]],
        **{"neur" + str(i): [["id", "not"], atomList] for i in range(maxSize)}
    }


def clauses_to_lambda(clauseList, atomList, maxSize):
    return lambda j: [atomList.index(atomKey) for atomKey in clauseList[j]] \
                     + [len(atomList) for i in range(maxSize - len(clauseList[j]))] + list(clauseList[j].values()) \
                     + [0 for i in range(maxSize - len(clauseList[j]))]


if __name__ == "__main__":
    testClause = [{"c": 0, "b": 1}, {"a": 0}]
    atomList = ["a","b","c"]
    maxSize = 2

    relCor = tnreason.representation.basis_calculus.create_relational_encoding(
        inshape=[len(testClause)],
        outshape=[len(atomList)+1 for i in range(maxSize)] + [2 for i in range(maxSize)],
        incolors=["l"],
        outcolors=["neur"+str(i)+"_p0_selVar" for i in range(maxSize)] + ["neur"+str(i)+"_actVar" for i in range(maxSize)],
        indicesToIndicesFunction=clauses_to_lambda(testClause, atomList, 2))

    relCores = tnreason.representation.basis_calculus.create_partitioned_relational_encoding(
        inshape=[len(testClause)],
        outshape=[len(atomList)+1 for i in range(maxSize)] + [2 for i in range(maxSize)],
        incolors=["l"],
        outcolors=["neur"+str(i)+"_p0_selVar" for i in range(maxSize)] + ["neur"+str(i)+"_actVar" for i in range(maxSize)],
        function=clauses_to_lambda(testClause, atomList, 2),
        partitionDict={"clause1":["neur0_p0_selVar","neur0_actVar"], "clause2":["neur1_p0_selVar","neur1_actVar"]})


    engine.draw_factor_graph(
        {**representation.create_architecture(get_cnf_architecture(maxSize=2, atomList=atomList)),
        **relCores})