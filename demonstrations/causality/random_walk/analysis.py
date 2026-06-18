import numpy as np

from tnreason import  engine

def get_marginal_matrix(cores, margVariables, dim):
    margs = np.empty((len(margVariables), dim))
    for i, margVar in enumerate(margVariables):
        margs[i] = engine.contract(cores, openColors=[margVar]).values
    return margs
