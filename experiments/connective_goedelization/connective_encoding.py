import numpy as np

def get_binary_inputs(order=2):
    """
    Gives a list of all coordinates of the order, starting with [1, 1, ..., 1] ending with [0, 0, ..., 0]
    """
    binary_combinations = []
    for i in range(2**order):
        booleanCoordinate = list(map(int, np.binary_repr(2**order-1-i)))
        booleanCoordinate = [0 for _ in range(order-len(booleanCoordinate))] + booleanCoordinate
        binary_combinations.append(booleanCoordinate)
    return binary_combinations

def encode_nary_connective(lamFunc, order=2):
    return int("".join([str(lamFunc(*args)[0]) for args in get_binary_inputs(order)]), 2)

if __name__ == "__main__":
    from tnreason.representation import connectives as con

    assert encode_nary_connective(con.get_connectives("and"), 2) == 8
    assert encode_nary_connective(con.get_connectives("eq"), 2) == 9

    from experiments.connective_goedelization import connective_coding as cen

    order2func = cen.decode_nary_connective(7, order=2)
    assert encode_nary_connective(order2func, 2) == 7

    for decNum in [0, 5, 58, 97, 103]:
        order3func = cen.decode_nary_connective(decNum, order=3)
        assert encode_nary_connective(order3func, order=3) == decNum