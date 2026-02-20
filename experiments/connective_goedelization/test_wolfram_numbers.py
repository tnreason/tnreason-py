from tnreason import representation

core = representation.get_boolean_computation_core(functionName="3", inColors = ["c1"], outColor="wolframOut")
assert core[1, 1] == 1
assert core[1, 0] == 1

core = representation.get_boolean_computation_core(functionName="3", inColors = ["c1", "c2"], outColor="wolframOut")
assert core[1, 0, 0] == 1
assert core[0, 0, 0] == 0
assert core[0, 1, 1] == 1
assert core[1, 1, 1] == 0