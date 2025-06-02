import numpy as np

from tnreason import engine, application


def generate_random_positive_tn(hyperedgeDict, shapeDict):
    return {edgeKey: engine.get_core()(values=np.random.random(size=[shapeDict[var] for var in hyperedgeDict[edgeKey]]),
                                       colors=hyperedgeDict[edgeKey], name=edgeKey) for edgeKey in hyperedgeDict}


if __name__ == "__main__":

    tN = generate_random_positive_tn(hyperedgeDict={
        "e1": ["a", "b"],
        "v1": ["a"],
        "v2": ["b"],
        "e2": ["a", "b", "c"]
    }, shapeDict={"a": 2, "b": 3, "c": 5})

    pass