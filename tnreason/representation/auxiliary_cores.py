import math

from tnreason.representation import suffixes as suf
from tnreason.representation import coordinate_calculus as cc


## Special cases of activation cores

def create_boolean_head(color, headType, weight=None, coreType=None, name=None):
    """
    Created the head core to a boolean variable (with dimension 2)
    CHANGE: Expfactor and uncertainty not boolean!
    """
    if headType == "truthEvaluation": # tBasis
        headFunction = lambda x: x
    elif headType == "falseEvaluation": # fBasis
        headFunction = lambda x: 1 - x
    elif headType == "expFactor":
        headFunction = lambda x: math.exp(weight * x)
    elif headType == "uncertaintyAsWeight":
        headFunction = lambda x: (1 - x) * (1 - weight) + x * weight
    else:
        raise ValueError("Headtype {} not understood!".format(headType))
    if name is None:
        name = color + suf.actCoreSuf
    return {name: cc.create_tensor_encoding([2], [color], headFunction, coreType=coreType,
                                         name=name)}