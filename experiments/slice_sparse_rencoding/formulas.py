from tnreason import engine


def binary_rencoding(valueIterator, headColor):
    return [(val, {**posDict, headColor: 1}) for val, posDict in valueIterator] + [(1, {headColor: 0})] + [
        (-val, {**posDict, headColor: 0}) for val, posDict in valueIterator]

conjunctionList = [(1, {"x": 1, "y": 1})]
equivalenceList = [(1, {"x": 1, "y": 1}), (1, {"x": 0, "y": 0})]

print(binary_rencoding(conjunctionList, "con"))
print(binary_rencoding(equivalenceList, "eq"))
print(binary_rencoding([(1, {"x": 1, "y": 1})], "(and_x_y)"))
