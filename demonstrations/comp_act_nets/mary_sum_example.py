from tnreason import engine
import math


def get_sum_tn(catDim, catOrder):
    return {"block0": engine.create_from_slice_iterator(
        shape=[catDim, 2, catDim, catDim],
        colors=[f"Y_{0}", f"Z_{0}", f"X_{0}", f"TX_{0}"],
        sliceIterator=[(1, {f"Y_{0}": (x + tx) % catDim, f"Z_{0}": math.floor((x + tx) / catDim),
                            f"X_{0}": x, f"TX_{0}": tx}) for x in range(catDim) for tx in range(catDim)]),
        **{f"middleBlock{catEnumerator}": engine.create_from_slice_iterator(
            shape=[catDim, 2, catDim, catDim, 2],
            colors=[f"Y_{catEnumerator}", f"Z_{catEnumerator}", f"X_{catEnumerator}", f"TX_{catEnumerator}",
                    f"Z_{catEnumerator - 1}"],
            sliceIterator=[
                (1, {f"Y_{catEnumerator}": (x + tx + z0) % catDim,
                     f"Z_{catEnumerator}": math.floor((x + tx + z0) / catDim),
                     f"X_{catEnumerator}": x, f"TX_{catEnumerator}": tx, f"fZ_{catEnumerator - 1}": z0}) for x
                in range(catDim) for tx in range(catDim) for z0 in range(2)]
        ) for catEnumerator in range(1, catOrder - 1)},
        **{f"block{catOrder - 1}": engine.create_from_slice_iterator(
            shape=[catDim, 2, catDim, catDim, 2],
            colors=[f"Y_{catOrder - 1}", f"Y_{catOrder}", f"X_{catOrder - 1}", f"TX_{catOrder - 1}",
                    f"Z_{catOrder - 2}"],
            sliceIterator=[
                (1, {f"Y_{catOrder - 1}": (x + tx + z0) % catDim, f"Y_{catOrder}": math.floor((x + tx + z0) / catDim),
                     f"X_{catOrder - 1}": x, f"TX_{catOrder - 1}": tx, f"Z_{catOrder - 2}": z0}) for x
                in range(catDim) for tx in range(catDim) for z0 in range(2)]
        )}}


def encode_numbers(firstNumber, secondNumber, catDim):
    return {**{f"X_{len(firstNumber) - 1 - i}_eC": engine.create_from_slice_iterator(shape=[catDim], colors=[
        f"X_{len(firstNumber) - 1 - i}"], sliceIterator=[(1, {f"X_{len(firstNumber) - 1 - i}": int(number)})]) for
               i, number in
               enumerate(firstNumber)},
            **{f"TX_{len(secondNumber) - 1 - i}_eC": engine.create_from_slice_iterator(shape=[catDim], colors=[
                f"TX_{len(secondNumber) - 1 - i}"], sliceIterator=[
                (1, {f"TX_{len(firstNumber) - 1 - i}": int(number)})]) for i, number in
               enumerate(secondNumber)}}


assert 1 == encode_numbers("0001", "0000", 10)["X_0_eC"][{"X_0": 1}]
assert 0 == encode_numbers("0001", "0000", 10)["X_0_eC"][{"X_0": 0}]

catDim = 10
catorder = 2

## Check: Exactly one output
assert 1 == int(engine.contract(coreDict={**get_sum_tn(catDim, catorder), **encode_numbers("00", "00", catDim)},
                                openColors=[f"Y_{k}" for k in range(catorder + 1)])[{"Y_2": 0, "Y_1": 0, "Y_0": 0}])
assert 1 == int(engine.contract(coreDict={**get_sum_tn(catDim, catorder), **encode_numbers("00", "00", catDim)},
                                openColors=[])[:])
catDim = 2
catorder = 2
assert 1 == int(engine.contract(coreDict={**get_sum_tn(catDim, catorder), **encode_numbers("10", "11", catDim)},
                                openColors=[f"Y_{k}" for k in range(catorder + 1)])[{"Y_2": 1, "Y_1": 0, "Y_0": 1}])
assert 1 == int(engine.contract(coreDict={**get_sum_tn(catDim, catorder), **encode_numbers("10", "11", catDim)},
                                openColors=[])[:])
