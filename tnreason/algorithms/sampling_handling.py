from tnreason.algorithms import energy_based_algorithms as eba
from tnreason.algorithms import forward_sampling as fs

gibbsMethodString = "gibbsSample"
energySamplingMethods = [gibbsMethodString]

forwardSamplingString = "ForwardSampling"
coreSamplingMethods = [forwardSamplingString]

def get_energy_based_sampler(energyDict, samplingMethod, **specDict):
    if samplingMethod == gibbsMethodString:
        return eba.EnergyGibbsSampleCore(energyDict=energyDict, **specDict)

def get_core_based_sampler(cores, samplingMethod, **specDict):
    if samplingMethod == forwardSamplingString:
        return fs.ForwardSampleCore(cores, **specDict)
