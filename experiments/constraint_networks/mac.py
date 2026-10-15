"""Persistent MAC search with local tensor contractions and reversible domains."""

from __future__ import annotations
from collections import deque
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.generalized_arc_consistency import _domain_indicator_core, _supported_domain


Domain = tuple[int, ...]
DomainDict = dict[str, Domain]


def _supported_domains(core, domains, tensors, contraction):
    if contraction is None:
        return tuple((color, _supported_domain(core, color, domains)) for color in core.colors)

    support, args = contraction
    if support.ndim == 1:
        messages = (support,)
    elif support.ndim == 2:
        left, right = core.colors
        messages = (support @ tensors[right], support.T @ tensors[left])
    else:
        messages = tuple(np.einsum(*arguments) for arguments in args)
    # A removed value occurs in no supported tuple, so all axes can be revised together.
    return tuple((color, tuple(value for value in domains[color] if message[value]))
                 for color, message in zip(core.colors, messages))


@dataclass(frozen=True)
class PropagationResult:
    """Propagation outcome and work counters."""
    contradiction: bool
    reason: str | None
    revisions: int
    queue_pops: int
    domain_reductions: int
    removed_values: int

    def __bool__(self) -> bool:
        return not self.contradiction


@dataclass(frozen=True)
class Statistics:
    propagation_calls: int
    revisions: int
    queue_pops: int
    queue_pushes: int
    domain_reductions: int
    removed_values: int


@dataclass(frozen=True)
class SolveResult:
    """Solution or root domains; UNSAT need not be detected at the root."""
    solved: bool
    unsatisfiable: bool
    assignment: dict[str, int] | None
    domains: DomainDict
    nodes: int
    root_contradiction: bool
    node_limit_reached: bool
    statistics: Statistics


class ConstraintNetworkState:
    """Incremental GAC with reversible domains and private constraint cores.

    Optional on_closure(state, changed_colors) returns cores after each GAC closure.
    """

    def __init__(self, constraint_network, domains: Mapping[str, Iterable[int]] | None = None):
        self.constraint_network = constraint_network

        # One snapshot per search; branches reuse these constraints and indices.
        self._dimensions = dict(constraint_network.dimension_dict)
        self._core_keys_by_color = dict(constraint_network.core_keys_by_color)
        self._cores = deepcopy(constraint_network.coresDict)

        supplied_domains = dict(domains or {})
        unknown_colors = set(supplied_domains).difference(self._dimensions)
        if unknown_colors:
            raise KeyError(f"Unknown domain colors: {', '.join(sorted(unknown_colors))}")

        self._domains = {color: self._normalize_domain(color, supplied_domains.get(color, range(dim)))
                         for color, dim in self._dimensions.items()}
        self._domain_tensors = {color: np.zeros(dim, dtype=bool) for color, dim in self._dimensions.items()}
        for color, domain in self._domains.items():
            self._domain_tensors[color].put(domain, True)

        self._contractions = {}
        for core_key, core in self._cores.items():
            self._cache_contraction(core_key, core)

        self._queue: deque[str] = deque()
        self._queued: set[str] = set()
        self._trail: list[tuple[str, Domain]] = []
        self._level_markers = []
        self._temporary_core_keys, self._added_core_keys = [], set()
        self.on_closure, self._changed_colors = None, set(self._dimensions)
        self.contradiction, self.contradiction_reason = False, None

        self._propagation_calls = self._revisions = self._queue_pops = self._queue_pushes = 0
        self._domain_reductions = self._removed_values = 0

        for color, domain in self._domains.items():
            if not domain:
                self._set_contradiction(f"Initial domain for {color} is empty.")
                break

        for core_key, core in self._cores.items():
            if not self.contradiction and not core.colors and core[{}] == 0:
                self._set_contradiction(f"Scalar constraint {core_key} has no support.")
            self._enqueue(core_key)

    @property
    def domains(self) -> Mapping[str, Domain]:
        """Current domains, read-only."""
        return MappingProxyType(self._domains)

    @property
    def domain_cores(self) -> Mapping[str, object]:
        """Export independent copies of the unary domain cores."""
        return MappingProxyType({color: _domain_indicator_core(color, domain, self._dimensions[color])
                                 for color, domain in self._domains.items()})

    @property
    def dimensions(self) -> Mapping[str, int]:
        return MappingProxyType(self._dimensions)

    @property
    def decision_level(self) -> int:
        return len(self._level_markers)

    @property
    def statistics(self) -> Statistics:
        return Statistics(self._propagation_calls, self._revisions, self._queue_pops,
                             self._queue_pushes, self._domain_reductions, self._removed_values)

    def add_cores(self, cores, *, reversible=False):
        """Snapshot additions on existing colors; optional branch additions undo on pop."""
        if self.decision_level and not reversible:
            raise RuntimeError("Cores can only be added at the root.")
        additions = ConstraintNetwork(deepcopy(dict(cores)), maintain_indices=True)
        if self._cores.keys() & additions.coresDict.keys():
            raise ValueError("Added core keys must be new.")
        if any(self._dimensions.get(color) != dim for color, dim in additions.dimension_dict.items()):
            raise ValueError("Added cores must use existing colors with matching dimensions.")
        for core in additions.coresDict.values():
            values = getattr(core, "values", None)
            if isinstance(values, np.ndarray) and tuple(values.shape) != tuple(core.shape):
                raise ValueError("Core values must match the declared shape.")
        self._cores.update(additions.coresDict)
        self._added_core_keys.update(additions.coresDict)
        if self.decision_level:
            self._temporary_core_keys.extend(additions.coresDict)
        for color, keys in additions.core_keys_by_color.items():
            self._core_keys_by_color[color] += keys
        for key, core in additions.coresDict.items():
            self._cache_contraction(key, core)
            if not self.contradiction and not core.colors and core[{}] == 0:
                self._set_contradiction(f"Scalar constraint {key} has no support.")
            self._enqueue(key)
        return self

    def snapshot_domains(self) -> DomainDict:
        return dict(self._domains)

    def push_level(self) -> int:
        """Save a reversible search level."""
        self._level_markers.append((len(self._trail), len(self._temporary_core_keys), tuple(self._queue),
                                    self.contradiction, self.contradiction_reason))
        return self.decision_level

    def pop_level(self) -> int:
        """Restore the latest search level."""
        if not self._level_markers:
            raise RuntimeError("Cannot pop MAC state below the root decision level.")

        trail_size, core_size, queue, self.contradiction, self.contradiction_reason = self._level_markers.pop()
        removed = set(self._temporary_core_keys[core_size:])
        for key in removed:
            for color in self._cores.pop(key).colors:
                self._core_keys_by_color[color] = tuple(key for key in self._core_keys_by_color[color] if key not in removed)
            self._contractions.pop(key, None)
        del self._temporary_core_keys[core_size:]
        while len(self._trail) > trail_size:
            color, old_domain = self._trail.pop()
            self._domains[color] = old_domain
            self._domain_tensors[color].fill(False)
            self._domain_tensors[color].put(old_domain, True)
            if self.on_closure is not None:
                self._changed_colors.add(color)

        self._queue = deque(queue)
        self._queued = set(queue)
        return self.decision_level

    def assign(self, color: str, value: int) -> bool:
        """Assign a value; eliminated values cause contradiction."""
        return self.restrict_domain(color, (value,))

    def restrict_domain(self, color: str, values: Iterable[int]) -> bool:
        """Intersect a domain and report whether it changed."""
        if color not in self._dimensions:
            raise KeyError(f"Unknown color: {color}")
        allowed = set(self._normalize_domain(color, values))
        revised = tuple(value for value in self._domains[color] if value in allowed)
        return self._set_domain(color, revised)

    def propagate(self) -> PropagationResult:
        """Reach GAC closure, then repeat for additions from an optional closure hook."""
        revisions, pops = self._revisions, self._queue_pops
        reductions, removed = self._domain_reductions, self._removed_values
        self._propagation_calls += 1

        while not self.contradiction:
            while self._queue and not self.contradiction:
                core_key = self._queue.popleft()
                self._queued.discard(core_key)
                self._queue_pops += 1
                for color, revised in _supported_domains(self._cores[core_key], self._domains, self._domain_tensors,
                                                         self._contractions.get(core_key)):
                    self._revisions += 1
                    self._set_domain(color, revised, source=core_key)
                    if self.contradiction:
                        break
            if self.contradiction or self.on_closure is None:
                break
            changed, self._changed_colors = self._changed_colors, set()
            cores = self.on_closure(self, changed)
            if cores:
                self.add_cores(cores, reversible=True)
            if not self._queue:
                break

        return PropagationResult(self.contradiction, self.contradiction_reason,
                                    self._revisions - revisions, self._queue_pops - pops,
                                    self._domain_reductions - reductions, self._removed_values - removed)

    def assignment(self, colors: Iterable[str] | None = None) -> dict[str, int]:
        selected = self._dimensions if colors is None else tuple(colors)
        return {color: self._domains[color][0] for color in selected if len(self._domains[color]) == 1}

    def is_solved(self, colors: Iterable[str] | None = None) -> bool:
        selected = self._dimensions if colors is None else tuple(colors)
        return not self.contradiction and all(len(self._domains[color]) == 1 for color in selected)

    def _normalize_domain(self, color: str, values: Iterable[int]) -> Domain:
        dimension = self._dimensions[color]
        values = tuple(values)
        for value in values:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"Assignment for {color} must be an integer, got {type(value).__name__}.")
        normalized = tuple(sorted({int(value) for value in values}))
        invalid = [value for value in normalized if value < 0 or value >= dimension]
        if invalid:
            raise ValueError(f"Values {invalid} are outside domain 0..{dimension - 1} for {color}.")
        return normalized

    def _set_domain(self, color: str, revised: Domain, source: str | None = None) -> bool:
        old_domain = self._domains[color]
        if revised == old_domain:
            return False
        if not set(revised).issubset(old_domain):
            raise ValueError(f"Domain update for {color} is not monotone: {old_domain} -> {revised}.")

        self._trail.append((color, old_domain))
        self._domains[color] = revised
        self._domain_tensors[color].fill(False)
        self._domain_tensors[color].put(revised, True)
        if self.on_closure is not None:
            self._changed_colors.add(color)
        self._domain_reductions += 1
        self._removed_values += len(old_domain) - len(revised)

        if not revised:
            reason = f"constraint {source}" if source is not None else "explicit restriction"
            self._set_contradiction(f"{reason} emptied domain {color}.")
            return True

        for core_key in self._core_keys_by_color[color]:
            if core_key != source:
                self._enqueue(core_key)
        return True

    def _enqueue(self, core_key: str) -> None:
        if core_key not in self._queued:
            self._queue.append(core_key)
            self._queued.add(core_key)
            self._queue_pushes += 1

    def _cache_contraction(self, key, core):
        values = getattr(core, "values", None)
        if isinstance(values, np.ndarray) and values.ndim < 32:
            # Boolean contractions avoid signed cancellation and support-count overflow.
            support = values if values.dtype == bool else values != 0
            axes = tuple(range(support.ndim))
            args = tuple((support, axes) + tuple(item for axis, color in enumerate(core.colors) if axis != idx
                         for item in (self._domain_tensors[color], (axis,))) + ((idx,),)
                         for idx in axes) if support.ndim > 2 else ()
            self._contractions[key] = support, args

    def _set_contradiction(self, reason: str) -> None:
        self.contradiction = True
        self.contradiction_reason = reason


class ConstraintNetworkSolver:
    """Depth-first MAC search for constraint networks."""

    def __init__(self, constraint_network, max_nodes: int = 100000):
        if max_nodes <= 0:
            raise ValueError("max_nodes must be positive.")
        self.constraint_network = constraint_network
        self.max_nodes = max_nodes
        self.nodes, self.node_limit_reached = 0, False

    def solve(self, domains: Mapping[str, Iterable[int]] | None = None, *, state=None) -> SolveResult:
        """Search from fresh domains, or continue a prepared root state."""
        if state is not None:
            if domains is not None or state.constraint_network is not self.constraint_network:
                raise ValueError("A prepared state must belong to this network and cannot be combined with domains.")
            if state.decision_level:
                raise RuntimeError("Search must start at the root.")
        self.nodes, self.node_limit_reached = 0, False
        if state is None:
            state = self.constraint_network.create_mac_state(domains=domains)
        state.propagate()
        solution = self._search(state)
        result_domains = ({color: (value,) for color, value in solution.items()}
                          if solution is not None else state.snapshot_domains())
        return SolveResult(solution is not None, solution is None and not self.node_limit_reached,
                              solution, result_domains, self.nodes, state.contradiction,
                              self.node_limit_reached, state.statistics)

    def _search(self, state: ConstraintNetworkState) -> dict[str, int] | None:
        if self.nodes >= self.max_nodes:
            self.node_limit_reached = True
            return None
        self.nodes += 1

        if state.contradiction:
            return None
        if state.is_solved():
            return state.assignment()

        color = self._select_color(state)
        for value in state.domains[color]:
            state.push_level()
            try:
                state.assign(color, value)
                if state.propagate():
                    solution = self._search(state)
                    if solution is not None:
                        return solution
                if self.node_limit_reached:
                    return None
            finally:
                state.pop_level()
        return None

    def _select_color(self, state: ConstraintNetworkState) -> str:
        domains = state.domains
        return min((color for color, domain in domains.items() if len(domain) > 1),
                   key=lambda color: (len(domains[color]), -len(state._core_keys_by_color.get(color, ())), color))
