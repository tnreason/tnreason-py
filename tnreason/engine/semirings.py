"""Semiring definitions for Tensor Network contraction.

A semiring (S, ⊕, ⊗, 0, 1) determines what quantity the contraction
computes without changing the network topology.  Swapping the semiring
converts the same factor graph into a different reasoning engine:

    SUM_PRODUCT  (+, ×)        →  probabilistic marginals / partition function
    MAX_PRODUCT  (max, ×)      →  MAP inference / Viterbi
    BOOLEAN      (max, min)    →  satisfiability / CSP feasibility
    TROPICAL     (min, +)      →  shortest path / minimum energy

Usage
-----
    from tnreason.engine.semirings import TROPICAL
    result = engine.contract(coreDict, openColors=[], semiring=TROPICAL)
"""

from __future__ import annotations
import functools
from dataclasses import dataclass, field
from typing import Callable
import numpy as np


# ── Semiring dataclass ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Semiring:
    """A commutative semiring (S, add, mul, zero, one).

    Attributes
    ----------
    name : human-readable label
    add  : binary numpy-compatible operation (the ⊕ operator)
    mul  : binary numpy-compatible operation (the ⊗ operator)
    zero : additive identity (annihilator for mul)
    one  : multiplicative identity
    """
    name: str
    add:  Callable
    mul:  Callable
    zero: float
    one:  float

    def reduce_add(self, arr: np.ndarray, axis: int) -> np.ndarray:
        """Marginalise ``arr`` over ``axis`` using the ⊕ operator."""
        slices = np.moveaxis(arr, axis, 0)
        return functools.reduce(self.add, slices)

    def reduce_mul(self, arrays: list[np.ndarray]) -> np.ndarray:
        """Element-wise product of a list of arrays using ⊗."""
        return functools.reduce(self.mul, arrays)

    def __repr__(self) -> str:
        return f"Semiring({self.name})"


# ── Standard semirings ────────────────────────────────────────────────────────

SUM_PRODUCT = Semiring(
    name="SumProduct",
    add=np.add,
    mul=np.multiply,
    zero=0.0,
    one=1.0,
)
"""Standard probabilistic semiring (+, ×).
Z = total weight; marginals computed by keeping variables open."""

MAX_PRODUCT = Semiring(
    name="MaxProduct",
    add=np.maximum,
    mul=np.multiply,
    zero=0.0,
    one=1.0,
)
"""MAP / Viterbi semiring (max, ×).
Z = maximum joint probability; traceback gives the MAP assignment."""

BOOLEAN = Semiring(
    name="Boolean",
    add=np.maximum,   # OR on {0,1} floats
    mul=np.minimum,   # AND on {0,1} floats
    zero=0.0,
    one=1.0,
)
"""Boolean semiring (∨, ∧) on {0, 1} ⊂ ℝ.
Z = 1 iff a satisfying assignment exists; 0 otherwise."""

TROPICAL = Semiring(
    name="Tropical",
    add=np.minimum,   # min-plus: add = min
    mul=np.add,       # min-plus: mul = +
    zero=np.inf,
    one=0.0,
)
"""Tropical / min-plus semiring (min, +).
Z = minimum cost (shortest path, minimum energy)."""

SEMIRING_REGISTRY: dict[str, Semiring] = {
    "SumProduct": SUM_PRODUCT,
    "MaxProduct": MAX_PRODUCT,
    "Boolean":    BOOLEAN,
    "Tropical":   TROPICAL,
}


def get_semiring(name: str) -> Semiring:
    """Return a registered semiring by name."""
    if name not in SEMIRING_REGISTRY:
        raise ValueError(
            f"Unknown semiring '{name}'. "
            f"Available: {list(SEMIRING_REGISTRY)}"
        )
    return SEMIRING_REGISTRY[name]
