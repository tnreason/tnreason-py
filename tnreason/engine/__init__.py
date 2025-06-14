from tnreason.engine.engine_visualization import draw_factor_graph

from tnreason.engine.contraction_handling import contract, sum_contract, normalize, EngineUser

from tnreason.engine.core_creation import get_core, create_from_slice_iterator, convert, create_random_core

def get_dimDict(coreDict):
    dimDict = {}
    for coreKey in coreDict:
        dimDict.update({color: coreDict[coreKey].shape[i] for i, color in enumerate(coreDict[coreKey].colors)})
    return dimDict