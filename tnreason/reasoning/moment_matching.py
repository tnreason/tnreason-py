import numpy as np

from tnreason import engine, representation


class MomentMatcher(engine.EngineUser):
    """
    Fits alternatingly local cores to reproduce local expected Statistics.
    Equals to coordinate descent with optimal steplength of the likelihood.
        * targetCores: Vector Cores storing the local expected statistics to be matched
        * networkCores: Static Cores shaping the basis
    Generalizes the weight estimation (which is the special case of leg dimension 2 and fitting of exponentiated first coordinate.

    ! TargetCore and HeadCores have same keys and cannot be put together into a coreDict !
    """

    def __init__(self, structureCores, targetCores, headCores=None, binaryMoments=True, **engineSpec):
        super().__init__(**engineSpec)

        self.structureCores = structureCores
        self.targetCores = targetCores
        self.headCores = headCores or {key: engine.create_trivial_core(
            name=key, shape=targetCores[key].shape, colors=targetCores[key].colors, coreType=self.coreType
        ) for key in targetCores}

        self.dimensionDict.update(engine.get_dimDict({**self.structureCores, **self.targetCores}))

        self.binaryMoments = binaryMoments
        if self.binaryMoments:
            self.weightDict = {key: [] for key in targetCores}

    def compute_satVector(self, tboCore, normation=True):
        if normation:
            return engine.normate(coreDict={**self.structureCores, **self.headCores},
                                  outColors=tboCore.colors, inColors=[],
                                  contractionMethod=self.contractionMethod,
                                  coreType=self.coreType, dimensionDict=self.dimensionDict
                                  )
        else:
            return engine.contract(coreDict={**self.structureCores, **self.headCores},
                                   openColors=tboCore.colors,
                                   contractionMethod=self.contractionMethod,
                                   coreType=self.coreType, dimensionDict=self.dimensionDict
                                   )

    def binary_matching_step(self, headCoreKey, normation=True):
        tboCore = self.headCores.pop(headCoreKey)
        satVector = self.compute_satVector(tboCore, normation=normation)
        function, weight = solve_binary_moment_equation(
            satVect=satVector,
            empVect=self.targetCores[headCoreKey]
        )
        self.headCores[headCoreKey] = representation.create_tensor_encoding(
            inshape=[self.dimensionDict[color] for color in tboCore.colors],
            incolors=tboCore.colors,
            function=function, name=headCoreKey, coreType=self.coreType
        )
        self.weightDict[headCoreKey].append(weight)

    def matching_step(self, headCoreKey, normation=True):
        tboCore = self.headCores.pop(headCoreKey)
        satVector = self.compute_satVector(tboCore, normation=normation)
        self.headCores[headCoreKey] = representation.create_tensor_encoding(
            inshape=[self.dimensionDict[color] for color in tboCore.colors],
            incolors=tboCore.colors,
            function=solve_moment_equation(
                satVect=satVector,
                empVect=self.targetCores[headCoreKey]
            ), name=headCoreKey, coreType=self.coreType
        )

    def alternating_matching(self, sweepNum=10):
        for _ in range(sweepNum):
            for headCoreKey in list(self.headCores.keys()):
                if self.binaryMoments:
                    self.binary_matching_step(headCoreKey)
                else:
                    self.matching_step(headCoreKey)


def find_common_nonzero(vect1, vec2):
    """
    Searching for a reference position for quotients in moment match.
    -> Should be first entry of binary vector storing variable satisfaction
    """
    for i in np.ndindex(vect1.shape):
        if vect1[i] > 0 and vec2[i] > 0:
            return i
    else:
        return -1


def solve_moment_equation(satVect, empVect):
    """
    Solves the local expected statistics matching of the subscribable cores
        * satVect: Marginal probability of color wrt alien cores
        * empVect: Desired marginal probability (expected statistics)
    To Do: Update, such that partition function constant and not one reference coordinate!
    Special case of binary: Included!
    """
    refPos = find_common_nonzero(satVect, empVect)
    if refPos == -1:
        print("Warning: Moments cannot be matched!")
        return lambda i: 1

    return lambda *i: (satVect[refPos] / empVect[refPos]) * (empVect[tuple(i)] / satVect[tuple(i)])


def solve_binary_moment_equation(satVect, empVect):
    if satVect[0] == 0 or satVect[1] == 0:
        return lambda *i: 1, 0
    return lambda *i: (satVect[0] / empVect[0]) * (empVect[tuple(i)] / satVect[tuple(i)]), np.log(
        (satVect[0] / satVect[1]) * (empVect[1] / empVect[0]))
