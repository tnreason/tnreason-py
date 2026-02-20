import tnreason.representation.coordinate_calculus
from tnreason import engine, representation

import numpy as np


class ALS:
    """
    Implements the alternating least squares
        * networkCores: Main tensor network to be optimized
        * importanceList: List of tuples containing
            - tensor network specifying a loss by contraction
            - weight specifying the importance in the loss
        * importanceColors: Specifying the shared colors of networkCores and importanceList networks
        * targetCores: Specifying the fitting target after contraction
        * trivialKeys: Specifying cores of singe coordinates, which contribute only factors
    """

    def __init__(self, networkCores, importanceColors=[], importanceList=[(1, {})],
                 contractionMethod=None, targetCores={}):
        self.networkCores = networkCores

        self.importanceColors = importanceColors
        self.importanceList = importanceList
        self.contractionMethod = contractionMethod

        # To ease case, where only one element in targetList
        self.targetCores = targetCores
        self.trivialKeys = []  # Keys with single position, trivial in the sense that they will not be updated

    def random_initialize(self, updateKeys, shapesDict={}, colorsDict={}):
        for updateKey in updateKeys:
            if updateKey in self.networkCores:
                upShape = self.networkCores[updateKey].values.shape
                upColors = self.networkCores[updateKey].colors
                self.networkCores.pop(updateKey)
            else:
                upShape = shapesDict[updateKey]
                upColors = colorsDict[updateKey]
            if np.prod(upShape) > 1:
                self.networkCores[updateKey] = engine.create_random_core(updateKey, upShape, upColors,
                                                                         randomEngine="NumpyUniform")
            else:
                self.trivialKeys.append(updateKey)
                self.networkCores[updateKey] = tnreason.representation.coordinate_calculus.create_trivial_core(updateKey, upShape, upColors)

    def alternating_optimization(self, updateKeys, sweepNum=10, computeResiduum=False):
        updateKeys = [key for key in updateKeys if key not in self.trivialKeys]
        if computeResiduum:
            residua = np.empty((sweepNum, len(updateKeys)))
        for sweep in range(sweepNum):
            for i, updateKey in enumerate(updateKeys):
                self.optimize_core(updateKey)
                if computeResiduum:
                    residua[sweep, i] = self.compute_residuum()
        if computeResiduum:
            return residua

    ## Functionality now in algorithm.optimization_handling!
    # def get_color_argmax(self, updateKeys):
    #     # ! Only working for vectors -> Can be replaced by .get_maximal_index of NumpyCore
    #     return {self.networkCores[key].colors[0]: np.argmax(np.abs(self.networkCores[key].values)) for key in
    #             updateKeys}

    def optimize_core(self, updateKey):
        ## Trivialize the core to be updated (serving as a placeholder)
        tbUpdated = self.networkCores.pop(updateKey)
        updateColors = tbUpdated.colors
        updateShape = tbUpdated.shape
        dimDict = {color: updateShape[i] for i, color in enumerate(tbUpdated.colors)}

        conOperator = engine.sum_contract(self.importanceList,
                                          backCores={**self.networkCores,
                                                     **copy_cores(self.networkCores, "_out", self.importanceColors)},
                                          openColors=updateColors + [updateColor + "_out" for updateColor in
                                                                     updateColors],
                                          dimensionDict=dimDict
                                          )
        conTarget = engine.sum_contract(self.importanceList,
                                        backCores={**self.targetCores, **self.networkCores},
                                        openColors=updateColors,
                                        dimensionDict=dimDict)

        # engine.draw_factor_graph({**self.networkCores,
        #                                             **copy_cores(self.networkCores, "_out", self.importanceColors)})

        resultDim = int(np.prod(updateShape))
        conOperator.reorder_colors(updateColors + [color + "_out" for color in updateColors])
        flattenedOperator = conOperator.values.reshape(resultDim, resultDim)
        flattenedTarget = conTarget.values.flatten()

        ## Update the core by solution of least squares problem
        solution, res, rank, s = np.linalg.lstsq(flattenedOperator, flattenedTarget, rcond=None)

        controlRes = np.linalg.norm(np.matmul(flattenedOperator, solution) - flattenedTarget)
        if controlRes > 0.00001:
            print("Remaining Gradient {} at prediction scale {}".format(controlRes, np.linalg.norm(
                np.matmul(flattenedOperator, solution))))

        self.networkCores[updateKey] = engine.get_core()(solution.reshape(updateShape), updateColors,
                                                         updateKey)

        # print("PAR", res, rank, self.importanceColors)
        # contractedValues = engine.contract({updateKey: self.networkCores[updateKey],
        #                             **{key : self.networkCores[key] for key in self.networkCores if key !=updateKey},
        #                                     **copy_cores({key : self.networkCores[key] for key in self.networkCores if key !=updateKey}, "_out", self.importanceColors)},
        #                                    openColors=updateColors).values
        # conTarget.reorder_colors(updateColors)
        # #conTarget = engine.contract({**{key : self.networkCores[key] for key in self.networkCores if key !=updateKey},
        # #                                    **copy_cores(self.networkCores, "_out", self.importanceColors)},
        # #                                   openColors=updateColors).values
        #
        # print(np.linalg.norm(contractedValues-conTarget.values), np.linalg.norm(contractedValues))

    def compute_residuum(self):
        prediction = engine.contract(contractionMethod=self.contractionMethod,
                                     coreDict=self.networkCores,
                                     openColors=self.importanceColors)
        target = engine.contract(contractionMethod=self.contractionMethod,
                                 coreDict=self.targetCores,
                                 openColors=self.importanceColors)
        prediction.reorder_colors(target.colors)
        return np.linalg.norm(prediction.values - target.values)


def copy_cores(coreDict, suffix, exceptionColors):
    returnDict = {}
    for key in coreDict:
        core = coreDict[key].clone()
        newColors = core.colors
        for i, color in enumerate(newColors):
            if color not in exceptionColors:
                newColors[i] = color + suffix
        core.colors = newColors
        returnDict[key + suffix] = core
    return returnDict
