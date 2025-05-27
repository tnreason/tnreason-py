from tnreason import engine

class SampleCoreBase(engine.EngineUser):
    """
    Sampling is done via iterators over the datapoints
    """
    def __init__(self, sampleNum = 1, startSlices = None, colors=[], **engineSpec):
        super().__init__(**engineSpec)
        ## StartSlice initalizer
        self.sampleNum = sampleNum # Not required any more! Just an alternative initialization of start slices
        self.startSlices = startSlices or [(1, dict()) for _ in range(self.sampleNum)]
        self.colors = colors

        # DimensionDict correction and usage
        if self.dimensionDict is None:
            self.dimensionDict = dict()

        if len(self.colors) == 0:
            self.colors = list(self.dimensionDict.keys())
        else:
            for color in self.colors:
                if color not in self.dimensionDict:
                    self.dimensionDict[color] = 2

        self.shape = [self.dimensionDict[color] for color in self.colors]

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

    def to_core(self, coreType="PandasCore"):
        return engine.convert(self, coreType)
