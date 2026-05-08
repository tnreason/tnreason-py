"""Variable Elimination with min-fill contraction ordering.

Replaces the naive global einsum for networks where treewidth is
moderate but the number of variables is large (e.g. 4x4 Sudoku).

Algorithm
---------
1. Build the factor graph: which factors (cores) contain which variables.
2. Repeatedly pick the variable whose elimination adds the fewest new
   edges between its neighbours (min-fill heuristic).
3. Multiply all factors containing that variable, then sum it out.
4. Replace them with a single new factor on the remaining variables.
5. Repeat until only open variables remain.

Complexity: O(N * D^(treewidth+1)) where treewidth is determined by
the elimination ordering.  For a 4x4 Latin square, treewidth ~ 4,
giving max intermediate size 4^5 = 1024 — tractable.
"""

from __future__ import annotations
from types import SimpleNamespace
import numpy as np
from tnreason.engine import subscript_creation as subc


class VariableEliminationContractor:
    """Exact contraction via variable elimination with min-fill ordering."""

    def __init__(self, coreDict: dict, openColors: list):
        from tnreason.engine.workload_to_numpy import NumpyCore
        self.NumpyCore = NumpyCore
        # Work on copies so the original coreDict is not mutated
        self.factors: dict[str, NumpyCore] = {
            k: v.clone() for k, v in coreDict.items()
        }
        self.openColors: set[str] = set(openColors)
        # Determine elimination order
        all_colors = self._all_colors()
        self.elim_order = self._min_fill_order(all_colors - self.openColors)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _all_colors(self) -> set[str]:
        colors: set[str] = set()
        for core in self.factors.values():
            colors.update(core.colors)
        return colors

    def _factors_containing(self, color: str) -> list[str]:
        return [k for k, core in self.factors.items() if color in core.colors]

    def _neighbours(self, color: str) -> set[str]:
        """Variables that share at least one factor with `color`."""
        nbrs: set[str] = set()
        for k in self._factors_containing(color):
            nbrs.update(self.factors[k].colors)
        nbrs.discard(color)
        return nbrs

    def _fill_edges(self, color: str, remaining: set[str]) -> int:
        """Number of new edges added if `color` is eliminated next."""
        nbrs = self._neighbours(color) & remaining
        n = len(nbrs)
        # Existing edges among nbrs
        existing = 0
        nbrs_list = list(nbrs)
        for i, u in enumerate(nbrs_list):
            for v in nbrs_list[i + 1:]:
                if any(u in self.factors[k].colors and v in self.factors[k].colors
                       for k in self.factors):
                    existing += 1
        total_possible = n * (n - 1) // 2
        return total_possible - existing

    def _min_fill_order(self, to_eliminate: set[str]) -> list[str]:
        """Greedy min-fill elimination ordering."""
        remaining = set(to_eliminate)
        order: list[str] = []
        while remaining:
            best = min(remaining, key=lambda c: self._fill_edges(c, remaining))
            order.append(best)
            remaining.remove(best)
        return order

    def _merge_factors(self, left_colors, left_values, right_colors, right_values):
        """Merge two factors and return the merged values and color order."""
        merged_colors = list(dict.fromkeys([*left_colors, *right_colors]))
        merged_dict = {
            "A": SimpleNamespace(colors=left_colors),
            "B": SimpleNamespace(colors=right_colors),
        }
        substr, order, _, resulting_color_order = subc.get_einsum_substring(
            merged_dict, merged_colors
        )
        tensors = {"A": left_values, "B": right_values}
        merged_values = np.einsum(substr, *[tensors[key] for key in order])
        return merged_values, [c for c in resulting_color_order if c in merged_colors]

    # ── Core elimination step ─────────────────────────────────────────────────

    def _eliminate(self, color: str) -> None:
        """Multiply all factors containing `color`, sum it out, replace."""
        keys = self._factors_containing(color)
        if not keys:
            return

        # Build combined factor via sequential pairwise einsum
        combined_colors: list[str] = []
        combined_values = None

        for key in keys:
            core = self.factors[key]
            if combined_values is None:
                combined_colors = list(core.colors)
                combined_values = core.values.copy()
            else:
                combined_values, combined_colors = self._merge_factors(
                    combined_colors, combined_values, core.colors, core.values
                )

        # Sum out `color`
        if color in combined_colors:
            ax = combined_colors.index(color)
            combined_values = combined_values.sum(axis=ax)
            combined_colors = [c for c in combined_colors if c != color]

        # Remove old factors, add new one
        new_key = f"_ve_{'_'.join(sorted(combined_colors))}"
        for key in keys:
            del self.factors[key]
        if combined_colors:  # non-scalar
            self.factors[new_key] = self.NumpyCore(
                values=combined_values, colors=combined_colors, name=new_key
            )
        else:
            # Scalar — store as a scalar factor (empty colors)
            self.factors[new_key] = self.NumpyCore(
                values=np.array(combined_values), colors=[], name=new_key
            )

    # ── Public contract ───────────────────────────────────────────────────────

    def contract(self):
        """Run VE and return the result as a NumpyCore."""
        for color in self.elim_order:
            self._eliminate(color)

        # Combine remaining factors (all over open colors only)
        remaining = list(self.factors.values())
        if not remaining:
            return self.NumpyCore(values=np.array(1.0), colors=[], name="ve_result")

        # Sequential pairwise merge of remaining factors
        result_colors = list(remaining[0].colors)
        result_values = remaining[0].values.copy()

        for core in remaining[1:]:
            result_values, result_colors = self._merge_factors(
                result_colors, result_values, core.colors, core.values
            )

        # Reorder to match openColors
        open_list = [c for c in result_colors if c in self.openColors]
        if set(open_list) != set(result_colors):
            # Sum out any unexpected residuals
            for c in list(result_colors):
                if c not in self.openColors:
                    ax = result_colors.index(c)
                    result_values = result_values.sum(axis=ax)
                    result_colors.remove(c)

        return self.NumpyCore(values=result_values, colors=result_colors, name="ve_result")
