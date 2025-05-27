def get_connectives(connectiveKey):
    if connectiveKey == "imp":
        return lambda a, b: [int(not a or b)]
    elif connectiveKey == "and":
        return lambda a, b: [int(a and b)]
    elif connectiveKey == "or":
        return lambda a, b: [int(a or b)]
    elif connectiveKey == "xor":
        return lambda a, b: [int(a ^ b)]
    elif connectiveKey == "eq":
        return lambda a, b: [int(a == b)]
    elif connectiveKey == "lpas":
        return lambda a, b: [int(a)]
    elif connectiveKey == "rpas":
        return lambda a, b: [int(b)]
    elif connectiveKey == "id":
        return lambda a: [int(a)]
    elif connectiveKey == "not":
        return lambda a: [int(not a)]

    ## Identification by Wolfram number
    elif connectiveKey.startswith("u"):  # Unary connective
        return decode_nary_connective(int(connectiveKey[1:]), order=1)
    elif connectiveKey.startswith("b"):  # Binary connective
        return decode_nary_connective(int(connectiveKey[1:]), order=2)
    elif connectiveKey.startswith("t"):  # Ternary connective
        return decode_nary_connective(int(connectiveKey[1:]), order=3)
    elif connectiveKey.startswith("q"):  # Quaternary connective
        return decode_nary_connective(int(connectiveKey[1:]), order=4)

    else:
        raise ValueError("Connective {} not implemented!".format(connectiveKey))


def decode_nary_connective(decNumber, order=2):
    binDigits = bin(decNumber)[2:]
    if len(binDigits) != 2 ** order:
        binDigits = "0" * (2 ** order - len(binDigits)) + binDigits
    return lambda *args: [int(binDigits[2 ** order - 1 - int("".join(map(str, args)), 2)])]


def get_unary_connective_selector(connectiveList):
    return lambda l, a: get_connectives(connectiveList[l])(a)


def get_binary_connective_selector(connectiveList):
    return lambda l, a, b: get_connectives(connectiveList[l])(a, b)

# def get_connective_selector(connectiveList):
#     if connectiveList[0] in ["imp","and","or","xor","eq"]:
#         return lambda l, a, b : get_connectives(connectiveList[l])(a,b)
#     else:
#         return lambda l, a: get_connectives(connectiveList[l])(a)
