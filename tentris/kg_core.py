from tnreason.engine import workload_to_tentris as wt

termCore = wt.TentrisTripleStoreTermCore()
termCore.load_rdf_data("/home/version1/tentris/bbb_generated.ttl")
termCore.adjust_interpretationsDict()

from tnreason import engine

polyCore = engine.convert(termCore, "PolynomialCore")
print([val for val in polyCore])