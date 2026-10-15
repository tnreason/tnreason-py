"""Optional Sudoku pair and locked-candidate summaries for tensor-based MAC."""
import numpy as np
from tnreason import engine
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from demonstrations.sudoku.constraints.sudoku_constraints import get_sudoku_constraints
from .naked_pairs import detect_naked_pairs
from .hidden_pairs import detect_hidden_pairs


class _SudokuClusters:
    def __init__(self, state, num, locked=False):
        if not isinstance(num, int) or num < 1:
            raise ValueError("num must be a positive integer Sudoku box size.")
        self.num, self.categories = num, get_sudoku_constraints(num)
        dimensions, atoms = state.dimensions, set().union(*self.categories.values())
        if any(dimensions.get(color) != num ** 2 for color in self.categories) or any(dimensions.get(atom) != 2 for atom in atoms):
            raise ValueError("Sudoku clusters require the standard categorical Sudoku variables and dimensions.")
        self.links = {frozenset(core.colors): key for key, core in state._cores.items() if len(core.colors) == 2}
        self.supports, self.contractions, self.summaries, self.motifs, self.by_color = {}, {}, {}, {}, {}
        if locked:
            for color, atoms in self.categories.items():
                if not color.startswith(("row_", "col_")):
                    continue
                boxes = {}
                for atom in atoms:
                    fields = atom.split("_")
                    boxes.setdefault(f"square_{fields[5]}_{fields[1]}_{fields[3]}", []).append(atom)
                for box, shared in boxes.items():
                    colors = color, box
                    for atom in shared:
                        for unit in colors:
                            if frozenset((atom, unit)) not in self.links:
                                raise ValueError(f"Missing Sudoku link between {atom} and {unit}.")
                    self.motifs[colors] = ([*colors, *shared], tuple(
                        frozenset(self.categories[unit].index(atom) for atom in shared) for unit in colors))
                    for unit in colors:
                        self.by_color.setdefault(unit, []).append(colors)

    def _summary(self, state, cluster, arity, kind):
        colors, atoms = cluster[:arity], cluster[arity:]
        key = f"sudoku_{kind}_" + "__".join(colors)
        if key in self.summaries:
            return key, self.summaries[key]
        selected = {}
        for atom in atoms:
            for color in colors:
                if atom not in self.categories[color]:
                    continue
                link = self.links.get(frozenset((atom, color)))
                if link is None:
                    raise ValueError(f"Missing Sudoku link between {atom} and {color}.")
                if link not in self.supports:
                    core = state._cores[link]
                    values = state._contractions[link][0] if link in state._contractions else np.array([
                        core[dict(zip(core.colors, idx))] != 0 for idx in np.ndindex(tuple(core.shape))
                    ]).reshape(core.shape)
                    self.supports[link] = engine.get_core("NumpyCore")(values=values, colors=core.colors)
                selected[link] = self.supports[link]
        # Contract only original links, never domains. Cache equal locked motifs locally.
        axes = {color: idx for idx, color in enumerate(colors + atoms)}
        fingerprint = tuple((tuple(axes[color] for color in core.colors), core.values.shape, core.values.tobytes())
                            for core in selected.values()) if kind == "locked" else None
        if fingerprint is None or fingerprint not in self.contractions:
            summary = engine.contract(selected, openColors=colors, contractionMethod="NumpyEinsum")
            summary.reorder_colors(colors)
            values = summary.values != 0
            if fingerprint is not None:
                self.contractions[fingerprint] = values
        else:
            values = self.contractions[fingerprint]
        self.summaries[key] = engine.get_core("NumpyCore")(values=values, colors=colors)
        return key, self.summaries[key]

    def pairs(self, state):
        if state.contradiction or state.is_solved():
            return {}
        return dict(self._summary(state, cluster, 4, "pair") for cluster in
                    detect_naked_pairs(state, self.num, verbose=False) + detect_hidden_pairs(state, self.num, verbose=False)
                    if "sudoku_pair_" + "__".join(cluster[:4]) not in state._cores)

    def __call__(self, state, changed):
        if state.contradiction or state.is_solved():
            return {}
        summaries, seen = {}, set()
        for color in sorted(changed & self.by_color.keys()):
            for colors in self.by_color[color]:
                if colors in seen or "sudoku_locked_" + "__".join(colors) in state._cores:
                    continue
                seen.add(colors)
                cluster, positions = self.motifs[colors]
                domains = tuple(state.domains[unit] for unit in colors)
                if any(len(domains[idx]) > 1 and all(value in positions[idx] for value in domains[idx])
                       and any(value not in positions[1 - idx] for value in domains[1 - idx]) for idx in (0, 1)):
                    key, core = self._summary(state, cluster, 2, "locked")
                    summaries[key] = core
        return summaries


def _sudoku_cluster_cores(state, num, pairs=True, locked=False):
    clusters = _SudokuClusters(state, num, locked=locked)
    if not state.propagate() or state.is_solved():
        return {}
    return {**(clusters.pairs(state) if pairs else {}), **(clusters(state, set(state.dimensions)) if locked else {})}


def prepare_sudoku_state(network, domains=None, num=3, *, pairs=True, locked=False):
    """Prepare root pairs and activate useful locked summaries after each GAC closure.

    Locked detection checks affected units only. Original-link contractions are
    cached; branch additions undo on backtracking. Pairs are selected at the root.
    Summaries contain no domains; prepare anew if original cores change.
    Continue with network.solve_mac(state=state), without passing domains again.
    """
    state = network.create_mac_state(domains)
    clusters = _SudokuClusters(state, num, locked=locked)
    if state.propagate() and not state.is_solved():
        if pairs:
            state.add_cores(clusters.pairs(state))
        if locked:
            state.on_closure = clusters
            state.propagate()
    return state


def add_sudoku_pair_clusters(network, domains=None, num=3):
    """Export the strengthened network; prefer prepare_sudoku_state for search.

    Supplied domains select clusters only; pass them again when solving this
    exported network. Original cores remain untouched.
    """
    state = network.create_mac_state(domains)
    return ConstraintNetwork({**network.coresDict, **_sudoku_cluster_cores(state, num)}, maintain_indices=network.maintain_indices)
