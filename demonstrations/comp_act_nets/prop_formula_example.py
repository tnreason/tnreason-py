from tnreason import engine, application

expressionsDict = {
    "f0" :  ["and", ["or", "X_0", "X_1"], ["not", "X_2"]]
}

engine.draw_factor_graph(
    application.create_cores_to_expressionsDict(expressionsDict)
)