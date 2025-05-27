defaultCoreType = "NumpyCore"

def get_core(coreType=None):
    if coreType is None:
        coreType = defaultCoreType
    if coreType == "NumpyCore":
        from tnreason.engine.workload_to_numpy import NumpyCore
        return NumpyCore
    elif coreType == "PolynomialCore":
        from tnreason.engine.polynomial_handling import PolynomialCore
        return PolynomialCore
    elif coreType == "PandasCore":
        from tnreason.engine.workload_to_pandas import PandasCore
        return PandasCore
    elif coreType == "HypertrieCore":
        from tnreason.engine.workload_to_tentris import HypertrieCore
        return HypertrieCore
    elif coreType == "TorchCore":
        from tnreason.engine.workload_to_torch import TorchCore
        return TorchCore
    elif coreType == "TensorFlowCore":
        from tnreason.engine.workload_to_tensorflow import TensorFlowCore
        return TensorFlowCore
    else:
        raise ValueError("Core Type {} not supported.".format(coreType))

def create_from_slice_iterator(shape, colors, sliceIterator, coreType=defaultCoreType, name="Iterator"):
    core = get_core(coreType)(values=None, colors=colors, name=name, shape=shape)
    for value, sliceDict in sliceIterator:
        core[sliceDict] = value
    return core

def convert(inCore, outCoreType=None):
    if outCoreType is None:
        outCoreType = defaultCoreType
    return create_from_slice_iterator(inCore.shape, inCore.colors, iter(inCore), coreType=outCoreType)

def create_random_core(name, shape, colors,
                       randomEngine="NumpyUniform"):  # Works only for numpy cores! (do not have a random engine else)
    from tnreason.engine.workload_to_numpy import np_random_core
    return np_random_core(shape, colors, randomEngine, name)