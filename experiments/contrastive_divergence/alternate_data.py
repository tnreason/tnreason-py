# def transform_datapoint(assignmentDict, sampler, **specDict):
#     return sampler.sample(startAssignment = assignmentDict)
#
# def transform_dataSet(dataCore, sampler):
#     for value, entry in dataCore:
#         transform_datapoint(assignmentDict)


class DataTransformer:
    def __init__(self, sampler, dataCore):
        self.sampler = sampler
        self.dataCore = dataCore

    def __iter__(self):
        self.iterator = iter(self.dataCore)
        return self

    def __next__(self):
        value, assignmentDict = next(self.iterator)
        return (value, self.sampler.draw_sample(startAssignment=assignmentDict))


if __name__ == "__main__":
    from tnreason.algorithms import energy_based_algorithms as eba

    from tnreason import knowledge, encoding

    dist = knowledge.HybridKnowledgeBase(weightedFormulas={"w1": ["imp", "a", "b", 5.23]})

    sampler = eba.EnergyGibbs(dist.get_energy_dict(), colors=dist.distributedVariables,
                              dimDict={color: 2 for color in dist.distributedVariables})
    print(sampler.colors)

    sampler.draw_sample(
        startAssignment={"a" + encoding.suf.atomicVariableSuffix: 0, "b" + encoding.suf.atomicVariableSuffix: 1},
    temperatureList=[1 for i in range(5)])
