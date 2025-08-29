import pandas as pd

from tnreason import application, reasoning, engine

import numpy as np


def get_energy_tn(covariances, means, productNum, riskAversion=1.0):
    """
    Prepares the energy tensor for the Markowitz portfolio optimization problem:
    - architecture: Tensor Network decomposition of a Formula Selecting Network for pairwise activation
    - parCore: Represents the covariances and the means
    """
    architecture = {
        "headNeuron": [["and"],
                       ["prod" + str(i) for i in range(productNum)],
                       ["prod" + str(i) for i in range(productNum)]]
    }
    parCore = engine.create_from_slice_iterator(colors=['headNeuron_p0_sV', 'headNeuron_p1_sV'],
                                                shape=[productNum, productNum],
                                                sliceIterator=[(riskAversion * covariances[i, j],
                                                                {'headNeuron_p0_sV': i, 'headNeuron_p1_sV': j}) for i in
                                                               range(productNum) for j in
                                                               range(productNum)] + [(-means[i], {'headNeuron_p0_sV': i,
                                                                                                  'headNeuron_p1_sV': i})
                                                                                     for i in range(productNum)]
                                                )
    return {"pCore": parCore, **application.create_architecture(architecture, headNeuronNames=["headNeuron"])}



def budget_base_measure(productNum, budget, smallerThan=False):
    """
    Prepares boolean tensor with 1 when the budget is exactly used, 0 otherwise
    """
    if smallerThan:
        iterator = [(1, {"prod" + str(i) + "_dV": indices[i] for i in range(productNum)})
                    for indices in np.ndindex(*[2] * productNum) if np.sum(indices) <= budget]
    else:
        iterator = [(1, {"prod" + str(i) + "_dV": indices[i] for i in range(productNum)})
                    for indices in np.ndindex(*[2] * productNum) if np.sum(indices) == budget]

    return engine.create_from_slice_iterator(colors=["prod" + str(i) + "_dV" for i in range(productNum)],
                                             shape=[2] * productNum,
                                             sliceIterator=iterator,
                                             coreType="NumpyCore"
                                             )


if __name__ == "__main__":
    productNum = 10
    budget = 3

    cov = pd.read_csv("./examples/generated/covariance_matrix.csv", index_col=0).to_numpy()
    means = pd.read_csv("./examples/generated/mean_vector.csv")["mean"].to_numpy()

    energyTN = get_energy_tn(cov, means, productNum, riskAversion=0.1)
    energy = engine.contract({**energyTN, "bmCore": budget_base_measure(productNum, budget, smallerThan=True)},
                             openColors=["prod" + str(i) + "_dV" for i in range(productNum)])

    print("Best configuration: {}".format(np.unravel_index(np.argmin(energy.values), energy.values.shape)))

