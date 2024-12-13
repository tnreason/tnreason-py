class SampleCoreBase:
    def __init__(self, **specDict):
        """
        Can pass the iteration of any core in startSliceIterator
        """
        self.sampleNum = specDict.get("sampleNum", 1)
        self.startSlices = specDict.get("startSlices", [(1, dict()) for i in range(self.sampleNum)])
        self.colors = specDict.get("colors", [])
        self.dimDict = specDict.get("dimDict", dict())

        if self.dimDict is None:
            self.dimDict = dict()

        if len(self.colors) == 0:
            self.colors = list(self.dimDict.keys())
        else:
            for color in self.colors:
                if color not in self.dimDict:
                    self.dimDict[color] = 2

        self.shape = [self.dimDict[color] for color in self.colors]

        self.coreType = specDict.get("coreType", None)
        self.contractionMethod = specDict.get("contractionMethod", None)

    def __iter__(self):
        self.startSliceIterator = iter(self.startSlices)
        return self

    def __next__(self):
        try:
            value, assignment = next(self.startSliceIterator)
            return (value, self.draw_sample(assignment))
        except StopIteration:
            raise StopIteration

    def draw_sample(self, startAssignment=dict()):
        return startAssignment

