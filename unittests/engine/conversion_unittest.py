import unittest

from tnreason import engine

tests = ["t1", "t2"]

shapeDict = {
    "t1": [2, 3, 4],
    "t2": [1, 1]
}

colorDict = {
    "t1": ["a", "b", "c"],
    "t2": ["d1", "e1K"]
}

coreTypeDict = {
    "t1": ["NumpyCore", "PolynomialCore", "PandasCore"],
    "t2": ["NumpyCore", "PolynomialCore", "PandasCore"],
}

iteratorDict = {
    "t1": [(1.25, {"a": 0, "b": 1, "c": 2}), (-2.25, {"b": 1})],
    "t2": []
}

checkDict = {
    "t1": [((0, 1, 2), -1), ((0, 1, 1), -2.25)],
    "t2": [((0, 0), 0)]
}


class IteratorConversionTest(unittest.TestCase):

    def test_creation(self):
        for testKey in tests:
            for coreType in coreTypeDict[testKey]:
                core = engine.create_from_slice_iterator(shape=shapeDict[testKey], colors=colorDict[testKey],
                                                         sliceIterator=iter(iteratorDict[testKey]), coreType=coreType)
                for ind, val in checkDict[testKey]:
                    self.assertEqual(core[ind], val)

    def test_conversion(self):
        for testKey in tests:
            for fromType in coreTypeDict[testKey]:
                for toType in coreTypeDict[testKey]:
                    core = engine.create_from_slice_iterator(shape=shapeDict[testKey], colors=colorDict[testKey],
                                                             sliceIterator=iter(iteratorDict[testKey]),
                                                             coreType=fromType)

                    toCore = engine.convert(core, toType)

                    for ind, val in checkDict[testKey]:
                        self.assertEqual(toCore[ind], val)

    def test_reduce_empty(self):
        for coreType in ["PolynomialCore"]:
            core = engine.create_from_slice_iterator(shape=[2, 19, 3, 12, 1], colors=["a2", "a1", "a3", "someC", "a"],
                                                     sliceIterator=iter([]), coreType=coreType)
            core.reduce_colors(["someC", "a2", "a"])
            assert core.colors == ["someC", "a2", "a"]
            assert core.shape == [12, 2, 1]