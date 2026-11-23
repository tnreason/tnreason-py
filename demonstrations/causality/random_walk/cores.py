from tnreason import engine
from demonstrations.causality import interventions as iv


def get_transitionIterator(t, dim=3, left_p=0.25, right_p=0.25, stay_p=0.5):
    return [(stay_p, {f"X_{t}": i, f"X_{t - 1}": i}) for i in range(dim)] + [
        (left_p, {f"X_{t}": dim - 1, f"X_{t - 1}": 0})] + [(left_p, {f"X_{t}": i - 1, f"X_{t - 1}": i}) for i in
                                                           range(1, dim)] + [
        (right_p, {f"X_{t}": 0, f"X_{t - 1}": dim - 1})] + [(left_p, {f"X_{t}": i + 1, f"X_{t - 1}": i}) for i in
                                                            range(dim - 1)]


def get_transitionCore(t, dim, left_p=0.25, right_p=0.25, stay_p=0.5, withIntervention=False):
    if withIntervention:
        return engine.create_from_slice_iterator(shape=[dim, dim, dim + 1],
                                                 colors=[f"X_{t}", f"X_{t - 1}", f"D_{t}"],
                                                 sliceIterator=iv.add_doVariable(f"D_{t}", f"X_{t}", dim,
                                                                                 get_transitionIterator(t, dim, left_p,
                                                                                                        right_p,
                                                                                                        stay_p)))
    else:
        return engine.create_from_slice_iterator(
            shape=[dim, dim],
            colors=[f"X_{t}", f"X_{t - 1}"],
            sliceIterator=get_transitionIterator(t, dim, left_p, right_p, stay_p)
        )


def get_startCore(t, startPos, dim):
    return engine.create_from_slice_iterator(
        shape=[dim],
        colors=[f"X_{t}"],
        sliceIterator=[(1, {f"X_{t}": startPos})],
    )


def get_randomWalkCores(T, dim, left_p=0.25, right_p=0.25, stay_p=0.5, withIntervention=False):
    return {**{f"transition_{t}": get_transitionCore(t, dim, left_p, right_p, stay_p, withIntervention=withIntervention) for t in
               range(1, T)}, f"start": get_startCore(0, 0, dim)}


def get_responseCore(t, varSuf="X_", dim=3):
    return engine.create_from_slice_iterator(
        colors=[varSuf + f"{t}", varSuf + f"{t - 1}", f"L_{t}"],
        shape=[dim, dim, 3],
        sliceIterator=[(1, {varSuf + f"{t}": i - 1, varSuf + f"{t - 1}": i, f"L_{t}": 0}) for i in range(1, dim)] + [
            (1, {varSuf + f"{t}": dim - 1, varSuf + f"{t - 1}": 0, f"L_{t}": 0})] + [
                          (1, {varSuf + f"{t}": i, varSuf + f"{t - 1}": i, f"L_{t}": 1}) for i in range(dim)] + [
                          (1, {varSuf + f"{t}": i + 1, varSuf + f"{t - 1}": i, f"L_{t}": 2}) for i in
                          range(dim - 1)] + [(1, {varSuf + f"{t}": 0, varSuf + f"{t - 1}": dim - 1, f"L_{t}": 2})]
    )


def get_twinned_network(T, dim):
    return {**{f"resReal_{t}": get_responseCore(t, varSuf="X_", dim=dim) for t in range(1, T)},
            **{f"resCount_{t}": get_responseCore(t, varSuf="TX_", dim=dim) for t in range(1, T)},
            f"start": get_startCore(0, 0, dim)}


if __name__ == "__main__":
    print(engine.contract(get_randomWalkCores(T=3, dim=4), openColors=[])[:])
