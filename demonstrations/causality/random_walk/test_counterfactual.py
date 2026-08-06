from demonstrations.causality.random_walk import analysis as an
from demonstrations.causality.random_walk import cores as cr
from tnreason import engine

T = 4
dim = 4


def select_responses(responseDict):
    return {"res_" + color: engine.create_from_slice_iterator(
        colors=[color],
        shape=[3],
        sliceIterator=[(1, {color: responseDict[color]})]
    ) for color in responseDict}


print(an.get_marginal_matrix({**cr.get_twinned_network(T=T, dim=dim), "start": cr.get_startCore(1, 0, dim),
                              **select_responses({f"L_{t}": 2 for t in range(1, T)})}, [f"X_{i}" for i in range(1, T)], dim))
