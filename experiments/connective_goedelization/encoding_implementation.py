from tnreason import encoding

coreDict = encoding.create_formulas_cores(
    {"test": ["u3", "a", 2]}
)

assert coreDict["(u3_a)_cCore"].values[1, 1] == 1
assert coreDict["(u3_a)_cCore"].values[0, 1] == 1

coreDict = encoding.create_formulas_cores(
    {"test": ["b3", "a", "b", 2]} # 0101
)

assert coreDict["(b3_a_b)_cCore"].values[0, 0, 1] == 1
assert coreDict["(b3_a_b)_cCore"].values[0, 0, 0] == 0
assert coreDict["(b3_a_b)_cCore"].values[1, 1, 0] == 1
assert coreDict["(b3_a_b)_cCore"].values[1, 1, 1] == 0