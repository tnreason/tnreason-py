import morph_kgc

basePath = "/Users/alexgoessmann/Documents/ENEXA/tnreason/version1/demonstrations/kg_creation/"

g_rdflib = morph_kgc.materialize(basePath+"specfiles/morph_config.ini")
g_rdflib.serialize(basePath+"generated/generated.ttl")