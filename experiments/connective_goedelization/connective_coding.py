"""
To be incorporated into encoding/connectives
Coding scheme: binDigits encode the coordinates, listed starting from the largest towards the smallest
Oriented on Wolfram number of cellular automata
"""


def decode_nary_connective(decNumber, order=2):
    binDigits = bin(decNumber)[2:]
    if len(binDigits) != 2 ** order:
        binDigits = "0" * (2 ** order - len(binDigits)) + binDigits
    return lambda *args: [int(binDigits[2 ** order - 1 - int("".join(map(str, args)), 2)])]


def get_unary_connective(decNumber):
    return decode_nary_connective(decNumber, order=1)


if __name__ == "__main__":
    ones = decode_nary_connective(3, order=1)
    assert ones(0) == [1] and ones(1) == [1]

    zeros = decode_nary_connective(0, order=1)
    assert zeros(0) == [0] and zeros(1) == [0]

    order3func = decode_nary_connective(103, order=3)
    assert order3func(1, 1, 1) == [0]
    assert order3func(1, 1, 0) == [1]
    assert order3func(1, 0, 1) == [1]
    assert order3func(1, 0, 0) == [0]
    assert order3func(0, 1, 1) == [0]
    assert order3func(0, 1, 0) == [1]
    assert order3func(0, 0, 1) == [1]
    assert order3func(0, 0, 0) == [1]
