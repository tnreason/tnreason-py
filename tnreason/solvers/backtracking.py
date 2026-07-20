"""Backtracking search for TN constraint networks.

Combines ForwardChaining (arc consistency) with randomised DPLL-style
branching.  Each branch adds a unit-vector basis core that fixes one
variable to one value, then re-runs forward chaining.

Reference: Goessmann et al. (tnreason-py/version2), adapted and
promoted from experiments/ to the main library.
"""

from __future__ import annotations
import numpy as np
from tnreason import engine
from tnreason.solvers.forward_chaining import ForwardChaining


def _dim_dict(coresDict: dict) -> dict[str, int]:
    dims: dict[str, int] = {}
    for core in coresDict.values():
        for i, color in enumerate(core.colors):
            dims[color] = core.shape[i]
    return dims


def _unit_core(color: str, value: int, dim: int):
    """One-hot basis vector core that fixes color=value."""
    return engine.create_from_slice_iterator(
        colors=[color],
        shape=[dim],
        sliceIterator=[(1, {color: value})],
    )


def backtrack(
    coresDict: dict,
    depth: int = 0,
    max_depth: int = 20,
    verbose: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[bool, dict | None]:
    """Single backtracking pass with forward chaining at each node.

    Parameters
    ----------
    coresDict : factor cores defining the CSP.
    depth     : current recursion depth (internal).
    max_depth : maximum depth before declaring failure.
    verbose   : print progress.
    rng       : numpy random generator for reproducible runs.

    Returns
    -------
    (success, result_coresDict)
        success=True and result_coresDict contains unary factors
        (one per variable) when a solution is found.
    """
    if rng is None:
        rng = np.random.default_rng()

    fc = ForwardChaining(coresDict)
    fc.run()

    if not fc.consistent:
        if verbose:
            print(f"{'  '*depth}[depth {depth}] inconsistent")
        return False, None

    dims = _dim_dict(fc.coresDict)

    if fc.fully_disentangled:
        if verbose:
            print(f"{'  '*depth}[depth {depth}] solution found")
        return True, fc.coresDict

    if depth >= max_depth:
        return False, None

    # Pick an unresolved variable (random for diversity with restarts)
    unresolved = [v for v in dims if v not in fc.disentangledNodes]
    var = rng.choice(unresolved)
    values = rng.permutation(dims[var])

    if verbose:
        print(f"{'  '*depth}[depth {depth}] branching on {var}, "
              f"{len(unresolved)} unresolved")

    for val in values:
        new_cores = {
            **fc.coresDict,
            f"_branch_{var}_{val}": _unit_core(var, int(val), dims[var]),
        }
        success, result = backtrack(
            new_cores, depth + 1, max_depth, verbose, rng
        )
        if success:
            return True, result

    return False, None


def extract_solution(result_coresDict: dict) -> dict[str, int]:
    """Extract the MAP assignment from a solved (fully disentangled) coresDict.

    Returns {color: argmax_value} for every variable that has a unary factor.
    Values are 0-indexed (as stored internally).
    """
    import numpy as np
    solution: dict[str, int] = {}
    for core in result_coresDict.values():
        if len(core.colors) == 1:
            color = core.colors[0]
            if color not in solution:
                solution[color] = int(np.argmax(core.values))
    return solution


def backtrack_with_restarts(
    coresDict: dict,
    max_depth: int = 10,
    max_restarts: int = 100,
    verbose: bool = False,
    seed: int | None = None,
) -> tuple[bool, dict | None, int]:
    """Backtracking with randomised restarts.

    Randomises variable and value ordering on each restart, allowing
    the search to escape dead ends caused by unlucky early choices.

    Returns
    -------
    (success, result_coresDict, restarts_used)
    """
    rng_seed = seed  # use different seed each restart
    for restart in range(max_restarts):
        if verbose:
            print(f"\n=== Restart {restart} ===")
        rng = np.random.default_rng(
            None if rng_seed is None else rng_seed + restart
        )
        success, result = backtrack(
            coresDict, depth=0, max_depth=max_depth, verbose=verbose, rng=rng
        )
        if success:
            return True, result, restart

    return False, None, max_restarts
