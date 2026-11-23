import numpy as np

from tnreason import  engine

def get_marginal_matrix(cores, margVariables, dim):
    margs = np.empty((len(margVariables), dim))
    for i, margVar in enumerate(margVariables):
        rawMarginals = engine.contract(cores, openColors=[margVar]).values

        margs[i] = 1/np.sum(rawMarginals) * rawMarginals
    return margs
