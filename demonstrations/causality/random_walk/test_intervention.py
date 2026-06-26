from demonstrations.causality.random_walk import analysis as an
from demonstrations.causality.random_walk import cores as cr
from tnreason import engine

T = 5
dim = 4

## Rung 1 : Association - Marginals, conditional probabilities

cores = cr.get_randomWalkCores(T=T, dim=dim, withIntervention=False)

print(an.get_marginal_matrix(cr.get_randomWalkCores(T=T, dim=dim, withIntervention=False), [f"X_{i}" for i in range(T)],
                             dim=dim))



conditionedVar = 2
conditionedState = 1
print(an.get_marginal_matrix({**cr.get_randomWalkCores(T=T, dim=dim, withIntervention=False),
    "condCore": engine.create_from_slice_iterator(
        colors=[f"X_{conditionedVar}"],
        shape=[dim],
        sliceIterator=[(1, {f"X_{conditionedVar}": conditionedState})])
}, [f"X_{i}" for i in range(T)],
    dim=dim))

## Rung 2 : Intervention
## With intervention
interventedVar = 2
interventedState = 1

print(an.get_marginal_matrix({**cr.get_randomWalkCores(T=T, dim=dim, withIntervention=True),
    "intCore": engine.create_from_slice_iterator(
        colors=[f"D_{interventedVar}"],
        shape=[dim + 1],
        sliceIterator=[(1, {f"D_{interventedVar}": interventedState})]
    )
}, [f"X_{i}" for i in range(T)], dim=dim))
