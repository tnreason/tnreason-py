from tnreason.engine import workload_to_tentris as wt

kgCore = wt.ht_from_rdf("/home/version1/tentris/bbb_generated.ttl")



from tnreason import engine

polyCore = engine.convert(kgCore, "PolynomialCore")
print([val for val in polyCore])
#print(kgCore.values[1407374883553337, 1407374883553338, 1697645953286153])