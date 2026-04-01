from experiments.constraint_networks import constraint_networks as cn
from tnreason import engine

## Test whether modus ponens works, i.e .implication "a \Rightarrow b" disentangled by knowing the premise "a"
modusPonensNet = cn.ConstraintNetwork(coresDict={
    "impCore": engine.create_from_slice_iterator(
        colors=["a", "b"],
        shape=[2, 2],
        sliceIterator=[(1, dict()), (-1, {"a": 1, "b": 0})]
    ),
    "aKnownCore": engine.create_from_slice_iterator(
        colors=["a"],
        shape=[2],
        sliceIterator=[(1, {"a": 1})]
    )
}
)
modusPonensNet.add_subcontraction(contractKeys=["impCore", "aKnownCore"], openColors=["a", "b"],
                                  dropKeys=["aKnownCore", "impCore"], contractedName="sumCore")
assert len(modusPonensNet.coresDict) == 1
modusPonensNet.split_edge("sumCore", colors0=["a"], colors1=["b"])
assert modusPonensNet.coresDict["sumCore_0"][{"a": 0}] == 0
assert modusPonensNet.coresDict["sumCore_0"][{"a": 1}] == 1
assert modusPonensNet.coresDict["sumCore_1"][{"b": 0}] == 0
assert modusPonensNet.coresDict["sumCore_1"][{"b": 1}] == 1