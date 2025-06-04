import unittest

from tnreason.representation import features as ft

from tnreason import representation


class HardFeatureTest(unittest.TestCase):
    def test_capture_hardConstraint(self):
        hardFeat = ft.HardPartitionFeature(featureColors=["bazant"], interpretationDict={"bazant": [2, 1.2, 5, 6]})
        trivCanparam = hardFeat.create_trivial_canParam()
        self.assertTrue(trivCanparam[1] == 1)

        constraintVec = representation.create_vanishing_core(shape=[4], name="bazantCon", colors=["bazant"])
        constraintVec[{"bazant": 2}] = 1

        upCan = hardFeat.local_adjustment(environmentMean=constraintVec)
        self.assertTrue(upCan[1] == 0)
        self.assertTrue(upCan[2] == 1)
