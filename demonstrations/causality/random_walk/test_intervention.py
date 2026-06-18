from demonstrations.causality.random_walk import analysis as an
from demonstrations.causality.random_walk import cores as cr
from tnreason import engine

T = 5
dim = 4

## Without intervention

print(an.get_marginal_matrix(cr.get_randomWalkCores(T=T, dim=dim, withIntervention=True), [f"X_{i}" for i in range(T)],
                             dim=dim))

## With intervention
interventedVar = 1
interventedState = 2

print(an.get_marginal_matrix({**cr.get_randomWalkCores(T=T, dim=dim, withIntervention=True),
    "intCore": engine.create_from_slice_iterator(
        colors=[f"D_{interventedVar}"],
        shape=[dim+1],
        sliceIterator=[(1, {f"D_{interventedVar}": interventedState})]
    )
}, [f"X_{i}" for i in range(T)], dim=dim))
