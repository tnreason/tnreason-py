import unittest

from tnreason.representation import features as ft

from tnreason import representation, engine


class HardFeatureTest(unittest.TestCase):
    def test_capture_hardConstraint(self):
        hardFeat = ft.HardPartitionFeature(featureColors=["bazant"], shape=[3])
        trivCanparam = hardFeat.find_neutral_canParam()
        self.assertTrue(trivCanparam[1] == 1)

        constraintVec = engine.create_from_slice_iterator(shape=[4], colors=["bazant"],
                                                 sliceIterator=[])
        constraintVec[{"bazant": 2}] = 1

        upCan = hardFeat.local_adjustment(environmentMean=constraintVec)
        self.assertTrue(upCan[1] == 0)
        self.assertTrue(upCan[2] == 1)
