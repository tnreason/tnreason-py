import numpy as np

from tnreason.engine import subscript_creation as subc
from tnreason.engine import core_base as cb

def np_random_core(shape, colors, randomEngine, name):
    if randomEngine == "NumpyUniform":
        return NumpyCore(values=np.random.random(size=shape), colors=colors, name=name)
    elif randomEngine == "NumpyNormal":
        return NumpyCore(values=np.random.normal(size=shape), colors=colors, name=name)
    else:
        raise ValueError("Random Engine {} not known for core creation!".format(randomEngine))


class NumpyCore(cb.TensorCore):
    coreType = "NumpyCore"

    def __init__(self, values=None, colors=None, name="NoName", shape=None):
        super().__init__(colors, name, shape)

        if values is None:  # Empty initialization based on shape
            self.values = np.zeros(shape=self.shape).astype(float)
        else:  # Initialization based on values
            self.values = np.array(values)
            self.shape = self.values.shape

        self.index = 0

    def __getitem__(self, item):
        if isinstance(item, dict):
            return self.values[tuple(item.get(color, None) for color in self.colors)]
        elif isinstance(item, slice) and item.start is None:
            """ Then full slize : and Core needs to be just a number """
            assert len(self.colors) == 0
            return float(self.values)
        else:
            return self.values[item]

    def __setitem__(self, sliceDict, value):
        """
        Adds a value onto the slice, not erasing the values before!
        """
        subscript = tuple([slice(None) if color not in sliceDict else sliceDict[color] for color in self.colors])
        onesShape = [self.shape[i] for i, color in enumerate(self.colors) if color not in sliceDict]
        self.values[subscript] = self.values[subscript] + value * np.ones(shape=onesShape)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < np.prod(self.shape):
            indexTuple = np.unravel_index(self.index, self.shape)
            value = self.values[indexTuple]
            self.index += 1
            return (value, {color: indexTuple[i] for i, color in enumerate(self.colors)})
        else:
            self.index = 0
            raise StopIteration

    def clone(self):
        return NumpyCore(self.values.copy(), self.colors.copy(), self.name)  # ! Shallow Copies?

    def contract_with(self, core2): # For usage in Corewise Contractor
        newColors = list(set(self.colors) | set(core2.colors))
        newShapes = [0 for _ in newColors]
        for i, color in enumerate(self.colors):
            newShapes[newColors.index(color)] = self.shape[i]
        for i, color in enumerate(core2.colors):
            newShapes[newColors.index(color)] = core2.shape[i]
        return NumpyEinsumContractor(coreDict={self.name: self, core2.name: core2}, openColors=newColors).contract()

    ## For Sampling
    def normalize(self):
        return NumpyCore(1 / np.sum(self.values) * self.values, self.colors, self.name)

    ## For ALS: Reorder Colors and summation
    def reorder_colors(self, newColors):
        self.values = np.einsum(subc.get_reorder_substring(self.colors, newColors), self.values)
        self.colors = newColors

    def __add__(self, otherCore):
        if set(self.colors) != set(otherCore.colors):
            raise ValueError("Colors of summands {} and {} do not match!".format(self.name, otherCore.name))
        else:
            self.reorder_colors(otherCore.colors)
            return NumpyCore(self.values + otherCore.values, self.colors, self.name)

    def __rmul__(self, scalar):
        self.values = scalar * self.values
        return self

    def slice_multiply(self, weight, sliceDict=dict()):
        subscript = tuple([slice(None) if color not in sliceDict else sliceDict[color] for color in self.colors])
        self.values[tuple(subscript)] = weight * self.values[tuple(subscript)]
        return self

    def get_slice(self, colorEvidenceDict={}):
        newValues = self.values[tuple(colorEvidenceDict.get(color, slice(None))
                                        for i, color in enumerate(self.colors))]
        newColors = [color for color in self.colors if color not in colorEvidenceDict]
        newShape = [self.shape[i] for i, color in enumerate(self.colors) if color not in colorEvidenceDict]
        return NumpyCore(values=newValues, colors=newColors, shape=newShape, name="Sliced_"+self.name)

    def get_argmax(self):
        return {self.colors[i]: maxPos for i, maxPos in
                enumerate(np.unravel_index(np.argmax(self.values.flatten()), self.values.shape))}

    def draw_sample(self, asEnergy=False, temperature=1):
        if asEnergy:
            distribution = np.exp(self.values * 1 / temperature).flatten()
        else:
            distribution = self.values.flatten()
        sample = np.unravel_index(
            np.random.choice(np.arange(np.prod(distribution.shape)), p=distribution / np.sum(distribution)),
            self.values.shape)
        return {color: sample[i] for i, color in enumerate(self.colors)}

class NumpyEinsumContractor:
    def __init__(self, coreDict={}, openColors=[], optimize='greedy'):
        self.coreDict = {key: coreDict[key].clone() for key in coreDict}
        self.openColors = openColors
        self.optimize = optimize

    def contract(self):
        substring, coreOrder, colorDict, colorOrder = subc.get_einsum_substring(self.coreDict, self.openColors)
        operands = [self.coreDict[key].values for key in coreOrder]
        result = (np.einsum(substring, *operands, optimize=self.optimize)
                  if len(operands) > 2 else
                  np.einsum(substring, *operands))
        return NumpyCore(
            result,
            [color for color in colorOrder if color in self.openColors])


def _expand_to(values: np.ndarray, current_colors: list, target_colors: list,
               dim_map: dict) -> np.ndarray:
    """Expand a factor's values from current_colors to target_colors.

    Reorders axes to match target_colors order and inserts size-1 axes for
    missing colors, then broadcasts to the full target shape.
    """
    # Step 1: reorder current axes to their order in target
    cur_in_target = [c for c in target_colors if c in current_colors]
    if cur_in_target != current_colors:
        perm = [current_colors.index(c) for c in cur_in_target]
        values = np.transpose(values, perm)

    # Step 2: build reshape with 1s inserted for missing colors
    reshape, cur_idx = [], 0
    for color in target_colors:
        if color in current_colors:
            reshape.append(values.shape[cur_idx])
            cur_idx += 1
        else:
            reshape.append(1)
    values = values.reshape(reshape)

    # Step 3: broadcast to full target shape
    target_shape = tuple(dim_map[c] for c in target_colors)
    return np.broadcast_to(values, target_shape)


class SemiringContractor:
    """Variable elimination with an arbitrary semiring (⊕, ⊗).

    Supports Sum-Product, Max-Product, Boolean, and Tropical semirings
    (and any user-defined Semiring).  Elimination order is the order in
    which non-open colors appear across cores (min-fill not yet applied;
    sufficient for chain and tree networks).

    Parameters
    ----------
    coreDict   : factor dictionary (NumpyCore values)
    openColors : colors to keep in the output ([] for scalar Z)
    semiring   : a Semiring instance from tnreason.engine.semirings
    """

    def __init__(self, coreDict: dict, openColors: list, semiring=None):
        if semiring is None:
            from tnreason.engine.semirings import SUM_PRODUCT
            semiring = SUM_PRODUCT
        self.factors: dict = {k: v.clone() for k, v in coreDict.items()}
        self.open_colors: set = set(openColors)
        self.semiring = semiring
        self._counter = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _dim_map(self) -> dict:
        dm = {}
        for core in self.factors.values():
            for i, c in enumerate(core.colors):
                dm[c] = core.shape[i]
        return dm

    def _keys_containing(self, color: str) -> list:
        return [k for k, f in self.factors.items() if color in f.colors]

    def _all_colors(self) -> list:
        seen, order = set(), []
        for core in self.factors.values():
            for c in core.colors:
                if c not in seen:
                    seen.add(c); order.append(c)
        return order

    def _eliminate(self, color: str, dim_map: dict) -> None:
        """Multiply all factors containing color, then marginalise it out."""
        keys = self._keys_containing(color)
        if not keys:
            return

        # Union of all colors across these factors
        all_cols: list = []
        for k in keys:
            for c in self.factors[k].colors:
                if c not in all_cols:
                    all_cols.append(c)

        # Broadcast each factor to all_cols and multiply
        product = np.full(
            tuple(dim_map[c] for c in all_cols),
            self.semiring.one,
            dtype=float,
        )
        for k in keys:
            core = self.factors[k]
            expanded = _expand_to(
                core.values.astype(float), core.colors, all_cols, dim_map
            )
            product = self.semiring.mul(product, expanded)

        # Marginalise color out
        ax = all_cols.index(color)
        marginal = self.semiring.reduce_add(product, ax)
        remaining = [c for c in all_cols if c != color]

        # Replace old factors with the new one
        self._counter += 1
        new_key = f"_sr{self._counter}_{''.join(sorted(remaining))}"
        for k in keys:
            del self.factors[k]
        self.factors[new_key] = NumpyCore(
            values=marginal,
            colors=remaining,
            name=new_key,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def contract(self) -> "NumpyCore":
        """Run variable elimination and return the result as a NumpyCore."""
        dim_map = self._dim_map()
        to_elim = [c for c in self._all_colors() if c not in self.open_colors]

        for color in to_elim:
            self._eliminate(color, dim_map)

        # Combine remaining factors (all over open colors)
        remaining = list(self.factors.values())
        if not remaining:
            return NumpyCore(values=np.array(self.semiring.one), colors=[], name="sr_result")

        dim_map = self._dim_map()
        all_open = [c for c in self._all_colors() if c in self.open_colors]

        product = np.full(
            tuple(dim_map[c] for c in all_open) if all_open else (),
            self.semiring.one,
            dtype=float,
        )
        for core in remaining:
            expanded = (_expand_to(core.values.astype(float), core.colors, all_open, dim_map)
                        if all_open else core.values.astype(float))
            product = self.semiring.mul(product, expanded)

        return NumpyCore(values=product, colors=all_open, name="sr_result")