"""
Example: Countdown Game https://github.com/Jiayi-Pan/TinyZero?tab=readme-ov-file
"""

from examples.countdown_game import creation as cr
from tnreason import engine

varDimDict = {"a": [-5, 10], "b": [-5, 10]}
varColorList = ["a", "b"]
pmCutOff = [-10, 20]
weightSuperList = [[-1, 1], [1, -1], [1, 1]]
targetNumber = 5

selCore = cr.create_sum_selector(["a", "b"], varDimDict, "a+-b", [-10, 20], weightSuperList, "a+-_sVar", "NumpyCore")
headCore = engine.create_basis_core("", shape=[pmCutOff[1] - pmCutOff[0]], colors=["a+-b"],
                                    numberTuple=[targetNumber - pmCutOff[0]])

"""
Possible Combinations to get 2 are the non vanishing coordinates of the possible combination contraction:
"""
possibleCombinations = engine.contract({"sCore": selCore, "hCore": headCore}, openColors=["a+-_sVar", "a", "b"])

print("NumberOfPossibilies to combine two numbers between -5 and 10 to the number {}:".format(targetNumber),
      engine.contract({"pCom": possibleCombinations}, openColors=[]).values)
