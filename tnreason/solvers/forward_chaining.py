"""Forward Chaining for Tensor Network constraint satisfaction.

Arc-consistency / unit-propagation in TN language.
Ported and cleaned up from tnreason-py/version2/experiments/constraint_networks.

Key insight: only matrix-vector contractions are used.  When a variable
has a known unary factor, it is contracted into every multi-variable factor
that touches it (keeping ALL colors open — no summation yet).  Then
``split_edge`` tests whether the result factorises as marginal_A ⊗ marginal_B.
If yes, the factor splits into two unary factors and propagation continues.
If no, the factor remains (updated, but still multi-variable).
"""

from __future__ import annotations
import numpy as np
from tnreason import engine
from tnreason.engine.core_comparison import cores_close as _cores_close


class ForwardChaining:
    """Propagates unary factors through the TN until a fixed point.

    After calling ``run()``:
      - ``consistent``        : False if any factor has zero total weight.
      - ``fully_disentangled``: True if every variable has only unary factors.
      - ``coresDict``         : updated factor dictionary.
    """

    def __init__(self, coresDict: dict):
        self.coresDict: dict = {k: v.clone() for k, v in coresDict.items()}
        self.consistent: bool = True

        self.nodesDict: dict[str, list[str]] = {}        # color → factor keys
        self.singleNodeEdges: dict[str, list[str]] = {}  # color → unary keys

        for key, core in self.coresDict.items():
            for color in core.colors:
                self.nodesDict.setdefault(color, []).append(key)
            if len(core.colors) == 1:
                c = core.colors[0]
                self.singleNodeEdges.setdefault(c, []).append(key)

        self.disentangledNodes: list[str] = [
            v for v in self.singleNodeEdges
            if set(self.nodesDict[v]) <= set(self.singleNodeEdges[v])
        ]

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> "ForwardChaining":
        queue = [v for v in self.singleNodeEdges if v not in self.disentangledNodes]
        while queue and self.consistent:
            self._propagate_node(queue.pop(), queue)
        for v in list(self.singleNodeEdges):
            self._summarize_node(v)
        return self

    @property
    def fully_disentangled(self) -> bool:
        return len(self.disentangledNodes) == len(self.nodesDict)

    def marginal(self, color: str) -> np.ndarray | None:
        """Normalised marginal distribution for ``color``, or None."""
        keys = self.singleNodeEdges.get(color)
        if not keys:
            return None
        # Combine all unary factors for this color
        vals = np.ones(self.coresDict[keys[0]].values.shape)
        for k in keys:
            if k in self.coresDict:
                vals = vals * self.coresDict[k].values
        s = vals.sum()
        return vals / s if s > 0 else vals

    # ── Internal ──────────────────────────────────────────────────────────────

    def _summarize_node(self, color: str) -> None:
        """Merge >1 unary factors for ``color`` into one.

        Also detects contradiction: if the combined factor is all-zero,
        the constraint network is inconsistent.
        """
        keys = [k for k in self.singleNodeEdges.get(color, [])
                if k in self.coresDict]
        if len(keys) <= 1:
            return
        name = color + "_summary"
        combined = engine.contract(
            {k: self.coresDict[k] for k in keys}, openColors=[color]
        )
        if float(combined.values.sum()) == 0:
            self.consistent = False
            return
        self.coresDict[name] = combined
        for k in keys:
            if k in self.coresDict and k != name:
                del self.coresDict[k]
            self.nodesDict[color] = [x for x in self.nodesDict[color] if x != k]
        self.singleNodeEdges[color] = [name]
        if name not in self.nodesDict.get(color, []):
            self.nodesDict.setdefault(color, []).append(name)

    def _propagate_node(self, color: str, queue: list[str]) -> None:
        """Contract the known unary factor for ``color`` into every
        multi-variable factor that contains ``color``, keeping ALL colors
        open.  Then test for independence (split_edge)."""
        self._summarize_node(color)
        unary_keys = self.singleNodeEdges.get(color, [])
        if not unary_keys:
            return
        unary_key = unary_keys[0]
        if unary_key not in self.coresDict:
            return

        for edge_key in list(self.nodesDict.get(color, [])):
            if edge_key in self.singleNodeEdges.get(color, []):
                continue
            if edge_key not in self.coresDict:
                continue

            factor = self.coresDict[edge_key]
            if color not in factor.colors:
                continue

            all_colors = factor.colors  # keep ALL colors open
            other_colors = [c for c in all_colors if c != color]

            # Update factor: contract unary in, keep all colors open
            updated = engine.contract(
                {edge_key: factor, unary_key: self.coresDict[unary_key]},
                openColors=all_colors,
            )
            total = float(updated.values.sum())

            if total == 0:
                self.consistent = False
                return

            self.coresDict[edge_key] = updated

            # Try independence split
            if len(other_colors) == 0:
                # Scalar — remove
                del self.coresDict[edge_key]
                for c in all_colors:
                    self.nodesDict[c] = [x for x in self.nodesDict.get(c, [])
                                         if x != edge_key]
                continue

            split = self._try_split(edge_key, color, other_colors, total)
            if split == "inconsistent":
                self.consistent = False
                return
            elif split:
                # Factor split: color got a new unary, other variables may too
                unary_name = color + "_" + edge_key
                self.singleNodeEdges.setdefault(color, []).append(unary_name)
                self.nodesDict[color] = [
                    x for x in self.nodesDict.get(color, []) if x != edge_key
                ]
                self.nodesDict.setdefault(color, []).append(unary_name)

                if len(other_colors) == 1:
                    other = other_colors[0]
                    self.singleNodeEdges.setdefault(other, []).append(edge_key)
                    if other not in self.disentangledNodes and other not in queue:
                        queue.append(other)
            # If not split: factor remains multi-variable (updated values kept)

        # Mark disentangled if no multi-variable factors remain for color
        remaining_multi = [
            k for k in self.nodesDict.get(color, [])
            if k not in self.singleNodeEdges.get(color, [])
               and k in self.coresDict
               and len(self.coresDict[k].colors) > 1
        ]
        if not remaining_multi and color not in self.disentangledNodes:
            self.disentangledNodes.append(color)

    def _try_split(
        self, edge_key: str, color_a: str, colors_b: list[str], total: float
    ) -> bool | str:
        """Test if factor factorises as marginal_a ⊗ marginal_b.

        Returns True (split), False (not independent), "inconsistent" (zero).
        On True: installs two replacement factors in coresDict.
        """
        core = self.coresDict[edge_key]
        core_a = engine.contract({edge_key: core}, openColors=[color_a])
        core_b = engine.contract({edge_key: core}, openColors=colors_b)

        # Product of marginals, normalised
        product = engine.contract(
            {"a": core_a, "b": core_b}, openColors=core.colors
        )
        # Normalise by total^2 / total = total to match scale
        # product[a,b] = core_a[a] * core_b[b]
        # original[a,b] = core[a,b]; sum = total
        # Independence: core[a,b] = core_a[a] * core_b[b] / total
        # So we compare (1/total)*product with core
        if not _cores_close(product, core, atol=1e-6 * total ** 2 / total):
            # Check normalised version
            prod_vals = product.values
            try:
                perm = [product.colors.index(c) for c in core.colors]
                prod_vals = np.transpose(prod_vals, perm)
            except (ValueError, IndexError):
                return False
            if not np.allclose(prod_vals / total, core.values, atol=1e-6):
                return False

        # Split succeeds — replace edge with two pieces
        unary_name = color_a + "_" + edge_key
        self.coresDict[unary_name] = core_a   # unary on color_a
        self.coresDict[edge_key]   = core_b   # remaining factor on colors_b

        # Update nodesDict for color_a
        self.nodesDict[color_a] = [
            x for x in self.nodesDict.get(color_a, []) if x != edge_key
        ]
        # nodesDict for other colors: edge_key now points to core_b (correct)
        return True
