from tnreason import engine

con2 = engine.create_from_slice_iterator(
    shape=[2,2],
    colors = ["Suspects","Locations"],
    sliceIterator=[(1,{}),(-1,{"Suspects": 0, "Locations": 1})]
)
con3 = engine.create_from_slice_iterator(
    shape=[2],
    colors = ["Locations"],
    sliceIterator=[(1,{"Locations": 1})]
)

possibleSuspects = engine.contract(coreDict={"con2":con2, "con3":con3}, openColors=["Suspects"])
print("Possible Suspects:",possibleSuspects.values)