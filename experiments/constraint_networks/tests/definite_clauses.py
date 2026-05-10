from experiments.constraint_networks import unit_propagation as fc
from tnreason import engine

## Example in Norvig Fig 7.16
clauseList = [
    (["P"], ["Q"]),
    (["L", "M"], ["P"]),
    (["B", "L"], ["M"]),
    (["A", "P"], ["L"]),
    (["A", "B"], ["L"]),
    ([], ["A"]),
    ([], ["B"])
]


def clause_to_core(neg, pos):
    return engine.create_from_slice_iterator(
        colors=neg + pos,
        shape=[2 for _ in neg + pos],
        sliceIterator=[(1, dict()), (-1, {**{col: 1 for col in neg}, **{col: 0 for col in pos}})]
    )


def clauses_to_coreDict(clauseList):
    return {"clause" + str(i): clause_to_core(colorTup[0], colorTup[1]) for i, colorTup in enumerate(clauseList)}


chainer = fc.GenericUnitPropagation(clauses_to_coreDict(clauseList))
chainer.propagate_all_singleNodeEdges()

## Check whether all
assert len(chainer.disentangledNodes) == 6

for col in ['B', 'A', 'L', 'M', 'P', 'Q']:
    assert chainer.cn.coresDict[col + "_core"][{col: 0}] == 0
    assert chainer.cn.coresDict[col + "_core"][{col: 1}] == 1
