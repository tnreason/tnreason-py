"""Standalone descriptions and bounded, exact Boolean-support projections.

Detection never contracts. Execution never interprets a Sudoku rule: it only
joins the original constraints named by the description and projects support.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from hashlib import sha256
from itertools import islice
from math import prod
from numbers import Integral
from types import MappingProxyType

import numpy as np

from demonstrations.sudoku.constraints.sudoku_constraints import get_sudoku_constraints
from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from tnreason.engine.workload_to_numpy import NumpyCore


def _finite_items(values, label, maximum):
    if isinstance(values, (str, bytes)) or not isinstance(values, Collection):
        raise TypeError(f"{label} must be a finite collection")
    if len(values) > maximum:
        raise ValueError(f"{label} exceeds its configured size bound")
    if isinstance(values, (set, frozenset)):
        return tuple(sorted(values, key=repr))
    return tuple(values)


@dataclass(frozen=True)
class RuleLimits:
    """All limits are per context/detection/execution, not global settings.

    Work counts dense support reads, sparse slice comparisons, and Boolean
    join/reduction entries. max_size bounds any individual input/intermediate,
    not total resident memory. Unary
    projections share atom elimination but otherwise count work cumulatively.
    """
    max_board_size: int = 16
    max_network_cores: int = 100_000
    max_domain_values: int = 1_000_000
    max_steps: int = 100_000
    max_patterns: int = 64
    max_subset_size: int = 4
    max_fish_size: int = 4
    max_chain_length: int = 8
    max_als_size: int = 3
    max_als_sets: int = 128
    max_petals: int = 4
    max_graph_nodes: int = 2048
    max_graph_edges: int = 16_384
    max_group_size: int = 3
    max_coloring_nodes: int = 32
    max_net_categories: int = 24
    max_net_radius: int = 3
    max_cores: int = 256
    max_work: int = 10_000_000
    max_planning_steps: int = 10_000_000
    max_size: int = 1_000_000
    max_output_size: int = 100_000
    max_cache_entries: int = 128

    def __post_init__(self):
        for name, value in vars(self).items():
            minimum = 0 if name == "max_cache_entries" else 1
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class RuleCluster:
    family: str
    candidate: SummaryClusterCandidate
    participating_variables: tuple[str, ...]
    internal_variables: tuple[str, ...]
    domain_inputs: tuple[str, ...]
    assumptions: tuple[str, ...] = ()
    evidence: tuple = ()

    @property
    def core_keys(self):
        return self.candidate.core_keys

    @property
    def open_colors(self):
        return self.candidate.open_colors


@dataclass(frozen=True)
class Detection:
    clusters: tuple[RuleCluster, ...]
    steps: int
    truncated: bool = False


class _Scan:
    def __init__(self, limits):
        self.limits, self.steps, self.truncated = limits, 0, False

    def step(self):
        if self.steps >= self.limits.max_steps:
            self.truncated = True
            return False
        self.steps += 1
        return True

    def finish(self, clusters, truncated=False):
        clusters = tuple(islice(iter(clusters), self.limits.max_patterns + 1))
        return Detection(clusters[:self.limits.max_patterns], self.steps,
                         self.truncated or truncated or len(clusters) > self.limits.max_patterns)


class RuleContext:
    """An immutable domain snapshot with bounded original-support caching.

    Accepts a ConstraintNetwork or original core mapping. This context does not
    run GAC or modify the network. Use a new context after changing domains.
    Original core values are read at execution time. Cache keys include realized
    support bytes so changing an original constraint cannot reuse stale support.
    """
    def __init__(self, network, domains=None, num=3, limits=None):
        self.limits = limits or RuleLimits()
        if isinstance(num, bool) or not isinstance(num, Integral) or num < 1:
            raise ValueError("num must be a positive integer box size")
        self.num, self.n = int(num), int(num) ** 2
        if self.n > self.limits.max_board_size:
            raise ValueError("Sudoku size exceeds max_board_size")
        if 6 * self.n ** 3 > self.limits.max_domain_values:
            raise ValueError("Sudoku domains exceed max_domain_values")
        source = getattr(network, "coresDict", network)
        if len(source) > self.limits.max_network_cores:
            raise ValueError("Network exceeds max_network_cores")
        self.cores = MappingProxyType(dict(source))
        categories = get_sudoku_constraints(self.num)
        self.categories = MappingProxyType({key: tuple(atoms) for key, atoms in categories.items()})
        dimensions, links = {}, {}
        for key, core in self.cores.items():
            colors, shape = tuple(core.colors), tuple(core.shape)
            if len(colors) != len(shape) or len(colors) != len(set(colors)):
                raise ValueError(f"Invalid core scope: {key}")
            for color, dimension in zip(colors, shape):
                if isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension < 1:
                    raise ValueError(f"Invalid dimension for {color}")
                if color in dimensions and dimensions[color] != dimension:
                    raise ValueError(f"Inconsistent dimension for {color}")
                dimensions[color] = int(dimension)
            if len(colors) == 2:
                links.setdefault(frozenset(colors), []).append(key)
        atoms = set().union(*categories.values())
        if any(dimensions.get(color) != self.n for color in categories) or any(
                dimensions.get(atom) != 2 for atom in atoms):
            raise ValueError("Expected standard categorical Sudoku variables and dimensions")
        self.dimensions = MappingProxyType(dimensions)
        self._links = links
        if sum(dimensions.values()) > self.limits.max_domain_values:
            raise ValueError("Declared dimensions exceed max_domain_values")
        if domains is not None and not isinstance(domains, Mapping):
            raise TypeError("domains must be a finite variable-to-domain mapping")
        supplied = dict(domains or {})
        if supplied.keys() - dimensions.keys():
            raise KeyError(f"Unknown domain variables: {sorted(supplied.keys() - dimensions.keys())}")
        normalized = {}
        remaining = self.limits.max_domain_values
        for color, dim in dimensions.items():
            if color in supplied:
                values = tuple(islice(iter(supplied[color]), remaining + 1))
                if len(values) > remaining:
                    raise ValueError("Supplied domain entries exceed max_domain_values")
                remaining -= len(values)
            else:
                values = tuple(range(dim))
            if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or not 0 <= v < dim
                   for v in values):
                raise ValueError(f"Invalid domain for {color}")
            normalized[color] = tuple(sorted(set(map(int, values))))
        self.domains = MappingProxyType(normalized)
        self._support_cache = OrderedDict()

    def pos(self, row, col):
        return f"pos_{row // self.num}_{row % self.num}_{col // self.num}_{col % self.num}"

    def atom(self, row, col, digit):
        return f"a_{row // self.num}_{row % self.num}_{col // self.num}_{col % self.num}_{digit}"

    def unit(self, kind, index, digit):
        if kind not in ("row", "col", "square"):
            raise ValueError(f"Unknown unit kind: {kind}")
        return f"{kind}_{digit}_{index // self.num}_{index % self.num}"

    def options(self, category):
        """Domain options compatible with its Boolean indicators' domains.

        This is a local domain view, not propagation through other categories.
        Detection is most useful on a caller-supplied GAC closure.
        """
        atoms = self.categories[category]
        forced = {idx for idx, atom in enumerate(atoms) if self.domains[atom] == (1,)}
        if len(forced) > 1:
            return frozenset()
        return frozenset(idx for idx in self.domains[category]
                         if 1 in self.domains[atoms[idx]] and (not forced or idx in forced)
                         and all(0 in self.domains[atom] for j, atom in enumerate(atoms) if j != idx))

    def scan(self):
        return _Scan(self.limits)

    def describe(self, family, categories, atoms=None, outputs=None, evidence=(), assumptions=()):
        """Name original incidence tensors; perform no numerical contraction.

        By default include indicators shared by at least two chosen categories.
        Explicit atoms may have degree one. All original links with the chosen
        scopes are included, even if a scope has several original tensors.
        """
        categories = tuple(dict.fromkeys(_finite_items(categories, "categories", len(self.categories))))
        assumptions = _finite_items(assumptions, "assumptions", self.limits.max_cores)
        evidence = _finite_items(evidence, "evidence", self.limits.max_domain_values)
        if not categories or set(categories) - self.categories.keys():
            raise ValueError("Descriptions require known categorical variables")
        counts = {}
        for category in categories:
            for atom in self.categories[category]:
                counts[atom] = counts.get(atom, 0) + 1
        if atoms is not None:
            atoms = _finite_items(atoms, "atoms", len(self.dimensions))
        selected_atoms = tuple(sorted(atom for atom, count in counts.items() if count >= 2)) if atoms is None else tuple(dict.fromkeys(atoms))
        if set(selected_atoms) - counts.keys():
            raise ValueError("Every indicator must belong to a selected category")
        keys = []
        for atom in selected_atoms:
            for category in categories:
                if atom in self.categories[category]:
                    link_keys = self._links.get(frozenset((category, atom)))
                    if not link_keys:
                        raise ValueError(f"Missing Sudoku link between {atom} and {category}")
                    keys.extend(link_keys)
        keys = tuple(dict.fromkeys(keys))
        if not keys:
            raise ValueError("Description contains no original links")
        participating = tuple(dict.fromkeys(color for key in keys for color in self.cores[key].colors))
        outputs = categories if outputs is None else _finite_items(outputs, "outputs", len(self.dimensions))
        if len(outputs) != len(set(outputs)) or set(outputs) - set(participating):
            raise ValueError("Outputs must be distinct participating variables")
        internal = tuple(color for color in participating if color not in outputs)
        fingerprint = repr((family, keys, outputs, tuple(assumptions))).encode()
        key = "sudoku_rule_" + str(family) + "_" + sha256(fingerprint).hexdigest()[:16]
        return RuleCluster(str(family), SummaryClusterCandidate(key, keys, outputs), participating,
                           internal, participating, tuple(assumptions), tuple(evidence))

    def describe_tensors(self, family, core_keys, outputs=(), *, evidence=(), assumptions=()):
        """Describe arbitrary *original* tensors, including encoded extra axioms.

        This provides an explicit proof-cluster interface for compound rules,
        arithmetic/variant constraints, or domain-specific premises. It never
        manufactures an axiom and never searches. Execution has the same hard
        budgets as incidence descriptions. A declared assumption without its
        actual original tensor remains unsupported.
        """
        core_keys = _finite_items(core_keys, "core_keys", self.limits.max_network_cores)
        assumptions = _finite_items(assumptions, "assumptions", self.limits.max_cores)
        evidence = _finite_items(evidence, "evidence", self.limits.max_domain_values)
        keys = tuple(sorted(set(core_keys)))
        if not keys or set(keys) - self.cores.keys():
            raise ValueError("Descriptions require existing original tensor keys")
        outputs = _finite_items(outputs, "outputs", len(self.dimensions))
        participating = tuple(dict.fromkeys(c for key in keys for c in self.cores[key].colors))
        if len(outputs) != len(set(outputs)) or set(outputs) - set(participating):
            raise ValueError("Outputs must be distinct participating variables")
        internal = tuple(c for c in participating if c not in outputs)
        signature = repr((family, keys, outputs, tuple(assumptions))).encode()
        key = "sudoku_rule_" + str(family) + "_" + sha256(signature).hexdigest()[:16]
        return RuleCluster(str(family), SummaryClusterCandidate(key, keys, outputs), participating,
                           internal, participating, tuple(assumptions), tuple(evidence))


@dataclass(frozen=True)
class ContractionResult:
    status: str
    summaries: dict
    supported_domains: dict
    contradiction: bool
    reason: str | None
    work: int
    max_size: int
    planning_steps: int = 0


class _BudgetExceeded(Exception):
    pass


class _Work:
    def __init__(self, limits):
        self.limits, self.work, self.max_size = limits, 0, 0
        self.planning_steps = 0

    def plan(self, steps):
        if self.planning_steps + steps > self.limits.max_planning_steps:
            raise _BudgetExceeded("contraction planning exceeds max_planning_steps")
        self.planning_steps += steps

    def charge(self, work, size):
        self.max_size = max(self.max_size, size)
        if size > self.limits.max_size:
            raise _BudgetExceeded("intermediate/input size exceeds max_size")
        if self.work + work > self.limits.max_work:
            raise _BudgetExceeded("Boolean contraction work exceeds max_work")
        self.work += work


@dataclass(frozen=True)
class _Factor:
    colors: tuple[str, ...]
    values: np.ndarray


def _support(context, key, budget):
    core = context.cores[key]
    shape = tuple(core.shape)
    if len(shape) > 31:
        raise _BudgetExceeded("input rank exceeds supported NumPy rank (31)")
    size = prod(shape)
    budget.charge(size, size)
    values = getattr(core, "values", None)
    if isinstance(values, np.ndarray):
        if values.shape != shape:
            raise ValueError(f"Core values do not match declared shape: {key}")
        realized = values != 0
    else:
        if getattr(core, "coreType", None) not in ("PolynomialCore", "PandasCore"):
            raise ValueError("No bounded support reader for tensor backend: "
                             + str(getattr(core, "coreType", type(core).__name__)))
        # These backends scan stored slices/rows for each realized entry. Charge
        # every comparison before invoking a getter, even for cancelled slices.
        budget.charge(size * max(1, len(values)) * max(1, len(shape)), size)
        # Read realized entries, including cancellations of overlapping sparse
        # slices. Do not consume or reset a core's stateful iterator.
        realized = np.array([core[dict(zip(core.colors, idx))] != 0
                             for idx in np.ndindex(shape)], dtype=bool).reshape(shape)
    fingerprint = (key, tuple(core.colors), shape, realized.tobytes())
    cache = context._support_cache
    if context.limits.max_cache_entries:
        if fingerprint in cache:
            realized = cache.pop(fingerprint)
        cache[fingerprint] = realized
        while len(cache) > context.limits.max_cache_entries:
            cache.popitem(last=False)
    return _Factor(tuple(core.colors), realized)


def _join_reduce(factors, eliminate, dimensions, budget):
    """Boolean AND join followed by existential OR, with explicit size checks."""
    colors = tuple(dict.fromkeys(c for factor in factors for c in factor.colors))
    shape = tuple(dimensions[c] for c in colors)
    size = prod(shape)
    output = tuple(c for c in colors if c not in eliminate)
    # NumPy has an implementation-dependent dimensionality cap. Singleton
    # compaction usually avoids this; skip explicitly instead of allocating.
    if len(shape) > 31:
        raise _BudgetExceeded("intermediate rank exceeds supported NumPy rank (31)")
    budget.charge(size * (len(factors) + bool(eliminate)), size)
    values = np.ones(shape, dtype=bool)
    for factor in factors:
        ordered = tuple(c for c in colors if c in factor.colors)
        permutation = tuple(factor.colors.index(c) for c in ordered)
        aligned = factor.values.transpose(permutation) if permutation else factor.values
        broadcast_shape = tuple(dimensions[c] if c in factor.colors else 1 for c in colors)
        values &= aligned.reshape(broadcast_shape)
    axes = tuple(i for i, c in enumerate(colors) if c in eliminate)
    if axes:
        values = np.any(values, axis=axes)
    return _Factor(output, values)


def _eliminate(factors, color, dimensions, budget):
    budget.plan(2 * len(factors))
    bucket = [factor for factor in factors if color in factor.colors]
    if not bucket:
        return factors
    remaining = [factor for factor in factors if color not in factor.colors]
    remaining.append(_join_reduce(bucket, {color}, dimensions, budget))
    return remaining


def _project(factors, outputs, dimensions, budget):
    """Greedy bucket elimination, minimizing joined scope before each step."""
    factors = list(factors)
    pending = set(c for factor in factors for c in factor.colors) - set(outputs)
    while pending:
        def cost(color):
            budget.plan(len(factors))
            scope = set(c for factor in factors if color in factor.colors for c in factor.colors)
            return (prod(dimensions[c] for c in scope), len(scope), color)
        color = min(pending, key=cost)
        factors = _eliminate(factors, color, dimensions, budget)
        pending.remove(color)
    result = _join_reduce(factors, set(), dimensions, budget)
    return result.values.transpose(tuple(result.colors.index(c) for c in outputs))


def contract_cluster(context, cluster, *, joint=False, use_domains=True):
    """Exactly project selected original support, without applying a rule.

    Default returns one unary summary for each output. ``joint=True`` produces
    their joint relation if its expanded output fits max_output_size. Domains
    compact every input axis and are therefore part of the result's scope:
    add these summaries reversibly in the same MAC branch, never in a sibling.
    ``use_domains=False`` creates a globally valid original-constraint summary.
    No domain-dependent result is cached. Assumptions are rejected explicitly;
    uniqueness/variant rules require a separately encoded original constraint.
    """
    if cluster.assumptions:
        raise ValueError("Unencoded extra assumptions are unsupported: " + ", ".join(cluster.assumptions))
    budget = _Work(context.limits)
    try:
        if len(cluster.core_keys) > context.limits.max_cores:
            raise _BudgetExceeded("selected tensor count exceeds max_cores")
        if not cluster.core_keys or len(set(cluster.core_keys)) != len(cluster.core_keys):
            raise ValueError("Cluster needs nonempty distinct original tensor keys")
        factors = [_support(context, key, budget) for key in cluster.core_keys]
        colors = tuple(dict.fromkeys(c for factor in factors for c in factor.colors))
        if (len(cluster.open_colors) != len(set(cluster.open_colors))
                or set(cluster.open_colors) - set(colors)):
            raise ValueError("Outputs must be distinct variables in selected original tensors")
        indices = {color: context.domains[color] if use_domains else tuple(range(context.dimensions[color]))
                   for color in colors}
        dimensions = {color: len(values) for color, values in indices.items()}
        # A zero-sized internal domain makes the whole cluster contradictory.
        if any(not values for values in indices.values()):
            groups = (cluster.open_colors,) if joint else tuple((c,) for c in cluster.open_colors)
            factors = None
        else:
            compacted = []
            for factor in factors:
                size = prod(dimensions[c] for c in factor.colors)
                budget.charge(size, size)
                values = factor.values[np.ix_(*(indices[c] for c in factor.colors))]
                fixed_axes = tuple(i for i, c in enumerate(factor.colors) if dimensions[c] == 1)
                if fixed_axes:
                    values = np.squeeze(values, axis=fixed_axes)
                compacted.append(_Factor(tuple(c for c in factor.colors if dimensions[c] != 1), values))
            factors = compacted
            constants = [factor for factor in factors if not factor.colors]
            if constants:
                budget.charge(len(constants), 1)
                constant = np.asarray(all(bool(factor.values) for factor in constants), dtype=bool)
                factors = [factor for factor in factors if factor.colors]
                factors.append(_Factor((), constant))
            # Eliminate local degree-one/two Boolean indicators immediately.
            # Higher-degree indicators use the verified greedy bucket path:
            # eliminating an internal position first can avoid a large clique
            # in template overlays, grouped implications and compound nets.
            for color in colors:
                if color.startswith("a_") and color not in cluster.open_colors:
                    budget.plan(len(factors))
                    if sum(color in factor.colors for factor in factors) <= 2:
                        factors = _eliminate(factors, color, dimensions, budget)
            groups = (cluster.open_colors,) if joint else tuple((c,) for c in cluster.open_colors)
        summaries, supported_domains, contradiction = {}, {}, False
        # Empty output requests still check scalar satisfiability.
        if not groups:
            groups = ((),)
        for group in groups:
            shape = tuple(context.dimensions[c] for c in group)
            if len(shape) > 31:
                raise _BudgetExceeded("output rank exceeds supported NumPy rank (31)")
            output_size = prod(shape)
            if output_size > context.limits.max_output_size:
                raise _BudgetExceeded("expanded output exceeds max_output_size")
            budget.charge(output_size, output_size)
            values = np.zeros(shape, dtype=bool)
            if factors is not None:
                active_outputs = tuple(c for c in group if dimensions[c] != 1)
                compact = _project(factors, active_outputs, dimensions, budget)
                compact = compact.reshape(tuple(dimensions[c] for c in group))
                values[np.ix_(*(indices[c] for c in group))] = compact
            contradiction |= not bool(np.any(values))
            suffix = "joint" if joint else (group[0] if group else "scalar")
            summaries[cluster.candidate.key + "__" + suffix] = NumpyCore(values=values, colors=list(group))
            for axis, color in enumerate(group):
                other_axes = tuple(i for i in range(len(group)) if i != axis)
                support = np.any(values, axis=other_axes) if other_axes else values
                supported_domains[color] = tuple(map(int, np.flatnonzero(support)))
        return ContractionResult("executed", summaries, supported_domains, contradiction, None,
                                 budget.work, budget.max_size, budget.planning_steps)
    except _BudgetExceeded as exc:
        # Never return partially executed projections as if a family completed.
        return ContractionResult("skipped_budget", {}, {}, False, str(exc), budget.work,
                                 budget.max_size, budget.planning_steps)
