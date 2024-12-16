from tnreason import engine


def estimate_bn(dataCores, parentsDict, contractionMethod, coreType):
    return {colorKey + "_estCore": engine.normate(dataCores, outColors=[colorKey], inColors=parentsDict[colorKey],
                                                  contractionMethod=contractionMethod, coreType=coreType) for colorKey in
            parentsDict}


def get_naive_parentsDict(featureColors=[], classColor=[]):
    return {**{color: [classColor] for color in featureColors},
            classColor: []}



