from tnreason.reasoning import energy_based_algorithms as eba
from tnreason.reasoning import forward_sampling as fs

gibbsMethodString = "gibbsSample"
energySamplingMethods = [gibbsMethodString]

forwardSamplingString = "forwardSampling"
coreSamplingMethods = [forwardSamplingString]

# def get_sampler(energyOrCore=dict(), samplingMethod=gibbsMethodString, **specDict):
#     if samplingMethod == gibbsMethodString:
#         return eba.EnergyGibbsSampleCore(energyDict=energyOrCore, **specDict)
#     elif samplingMethod == forwardSamplingString:
#         return fs.ForwardSampleCore(coreDict=energyOrCore, **specDict)


def get_energy_based_sampler(energyDict=dict(), samplingMethod=gibbsMethodString, **specDict):
    if samplingMethod == gibbsMethodString:
        return eba.EnergyGibbsSampleCore(energyDict=energyDict, **specDict)


def get_core_based_sampler(cores=dict(), samplingMethod=forwardSamplingString, **specDict):
    if samplingMethod == forwardSamplingString:
        return fs.ForwardSampleCore(cores, **specDict)
