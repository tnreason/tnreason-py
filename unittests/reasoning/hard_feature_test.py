from tnreason.reasoning import features as ft

from tnreason import representation

hardFeat = ft.HardPartitionFeature(featureColors=["bazant"], interpretationDict={"bazant" : [2,1.2,5,6]})

trivCanparam = hardFeat.create_trivial_canParam()

assert trivCanparam[1] == 1


constraintVec = representation.create_vanishing_core(shape=[4], name="bazantCon", colors=["bazant"])
constraintVec[{"bazant": 2}] = 1

upCan = hardFeat.local_adjustment(environmentMean=constraintVec)
assert upCan[1] == 0
assert upCan[2] == 1