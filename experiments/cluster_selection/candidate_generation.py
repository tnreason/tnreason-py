"""Bounded structural proposals, with optional domain-compacted cost estimates."""
from collections import deque
from hashlib import sha1
from itertools import zip_longest
from math import prod
from numbers import Integral
from types import SimpleNamespace
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from .candidate_evaluation_tnreason import SummaryClusterCandidate
from .contraction_cost import estimate_candidate_cost


class ClusterCandidateGenerator:
    """Reuse core/color indices to propose cycles and connected expansions.

    Accept a ConstraintNetwork or core dictionary. No cores are contracted or
    mutated. Recreate after changing core keys, colors, or shapes. Costs use
    actual dense dimensions by default; compact_domains opts into sliced axes.
    Default outputs favor non-Boolean colors; open_colors overrides this pool.
    Traversal and list sizes are bounded, so proposals are not exhaustive.

    Usage: generator = ClusterCandidateGenerator(network)
           candidates = generator.generate(state.domains)
    Evaluate candidates separately; this does not attach cores to MAC or train
    a predictor. The cost planner exposes a path; an evaluator must execute that
    path for its work estimate to describe the executed contraction order.
    """

    def __init__(self, network, *, max_cores=10, max_open_colors=4, max_cycle_length=8,
                 max_candidates=32, max_work=1_000_000, max_size=100_000, max_steps=8192):
        limits = (max_cores, max_open_colors, max_cycle_length, max_candidates, max_work, max_size, max_steps)
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 1 for value in limits):
            raise ValueError("Generator limits must be positive integers")
        if max_cycle_length < 2:
            raise ValueError("max_cycle_length must be at least two")
        self.max_cores, self.max_open_colors, self.max_cycle_length = max_cores, max_open_colors, max_cycle_length
        self.max_candidates, self.max_work, self.max_size, self.max_steps = max_candidates, max_work, max_size, max_steps
        self.cores = dict(getattr(network, "coresDict", network))
        indexed = ConstraintNetwork(self.cores, maintain_indices=True)
        self.dimensions, self.by_color = indexed.dimension_dict, indexed.core_keys_by_color
        if any(isinstance(dim, bool) or not isinstance(dim, Integral) or dim < 1 for dim in self.dimensions.values()):
            raise ValueError("Core dimensions must be positive integers")
        self.scopes = {key: frozenset(core.colors) for key, core in self.cores.items()}
        self.neighbors = {color: tuple(sorted((key, other) for key in keys for other in self.scopes[key] if other != color))
                          for color, keys in self.by_color.items()}
        self._costs = {}

    def generate_nested(self, domains=None, *, max_candidates=None, **options):
        """Preserve earlier proposals while doubling generation effort.

        Stages request 32, 64, ... candidates through the first power of two
        covering the requested count. max_steps is the largest stage's budget;
        earlier budgets scale proportionally. Matching stage budgets preserve
        prefixes across different requested counts, including counts below 32.
        """
        limit = self.max_candidates if max_candidates is None else max_candidates
        if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 0:
            raise ValueError("max_candidates must be a nonnegative integer")
        if not limit:
            return ()
        last = 32
        while last < limit:
            last *= 2
        previous_steps, quota, result, seen = self.max_steps, 32, [], set()
        try:
            while quota <= last:
                self.max_steps = max(1, previous_steps * quota // last)
                for candidate in self.generate(domains, max_candidates=quota, **options):
                    signature = candidate.core_keys, candidate.open_colors
                    if signature not in seen:
                        seen.add(signature)
                        result.append(candidate)
                    if len(result) >= quota:
                        break
                quota *= 2
        finally:
            self.max_steps = previous_steps
        return tuple(result[:limit])

    def generate(self, domains=None, *, changed_colors=None, open_colors=None, max_candidates=None,
                 cyclic_only=False, compact_domains=False):
        """Return independent candidates, without evaluating their pruning gain.

        Supplied domains may be partial. Singleton colors do not act as
        traversal bridges. changed_colors restricts seed colors, while optional
        open_colors names eligible outputs (not necessarily all retained).
        Distinct cycle lengths, trees, and expanded loops share the list budget.
        cyclic_only uses broader, distance-pruned search and diverse cycles or
        joined cycles, without trees or dangling tree expansions.
        compact_domains estimates sliced input axes, while the temporary output
        must still fit its original dimensions for GAC.
        """
        limit = self.max_candidates if max_candidates is None else max_candidates
        if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 0:
            raise ValueError("max_candidates must be a nonnegative integer")
        supplied = dict(domains or {})
        changed = set(self.dimensions if changed_colors is None else changed_colors)
        eligible = set(open_colors) if open_colors is not None else {
            color for color, dim in self.dimensions.items() if dim > 2}
        if open_colors is None and not eligible:
            eligible = set(self.dimensions)
        unknown = (supplied.keys() | changed | eligible) - self.dimensions.keys()
        if unknown:
            raise KeyError(f"Unknown colors: {', '.join(sorted(unknown))}")
        sizes = {}
        for color, dim in self.dimensions.items():
            values = tuple(supplied.get(color, range(dim)))
            if any(isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < dim for value in values):
                raise ValueError(f"Invalid domain for {color!r}")
            sizes[color] = len(set(values))
        if not limit or not eligible or 0 in sizes.values():
            return ()
        active = {color for color, size in sizes.items() if size > 1}
        nearby = {color: sum(1 - sizes[other] / self.dimensions[other]
                             for _, other in self.neighbors[color] if other in active) for color in active}
        pressure = {color: sum(nearby[other] for _, other in self.neighbors[color] if other in active)
                    for color in active}

        def rank(color):
            return (color not in changed, color not in eligible, sizes[color] <= 1,
                    sizes[color], -nearby.get(color, 0), -pressure.get(color, 0), color)

        anchors = sorted(active & changed, key=rank)[:max(8, limit)]
        if not anchors:
            return ()
        adjacency = {color: sorted(((key, other) for key, other in self.neighbors[color] if other in active),
                                   key=lambda edge: (rank(edge[1]), edge[0])) for color in active}
        families, steps = [], 0
        lengths = range(2, min(self.max_cores, self.max_cycle_length) + 1)
        # Separate length budgets prevent plentiful short cycles hiding longer ones.
        quota = self.max_steps // (2 * max(1, len(lengths)))
        seeds = anchors if cyclic_only else anchors[:8]
        distances = {}
        if cyclic_only:
            for start in seeds:
                distance, queue = {start: 0}, deque((start,))
                while queue:
                    color = queue.popleft()
                    if distance[color] >= self.max_cycle_length:
                        continue
                    for _, other in adjacency[color]:
                        if other not in distance:
                            distance[other] = distance[color] + 1
                            queue.append(other)
                distances[start] = distance
        for length in lengths:
            found = set()
            for start in seeds:
                budget = self.max_steps * length // (2 * sum(lengths)) if cyclic_only else quota
                remaining = budget // len(seeds)
                stack = [(start, (), frozenset((start,)))]
                while stack and remaining:
                    color, keys, visited = stack.pop()
                    closing = len(keys) + 1 == length
                    for key, other in reversed(adjacency[color]):
                        if closing and other != start:
                            continue
                        if cyclic_only and (key in keys or (other in visited and other != start)
                                            or distances[start].get(other, length + 1) > length - len(keys) - 1):
                            continue
                        if not remaining:
                            break
                        remaining -= 1
                        steps += 1
                        if key in keys:
                            continue
                        extended = keys + (key,)
                        if other == start and len(extended) == length:
                            found.add(tuple(sorted(extended)))
                        elif len(extended) < length and other not in visited:
                            stack.append((other, extended, visited | {other}))
            families.append(found)

        def scope(keys):
            return frozenset().union(*(self.scopes[key] for key in keys))

        def priority(keys):
            return (min(rank(color) for color in scope(keys)), len(keys), keys)

        cycle_seeds = (list(dict.fromkeys(keys for row in zip_longest(*(sorted(family, key=priority) for family in families))
                                         for keys in row if keys is not None)) if cyclic_only else
                       sorted(set().union(*families), key=priority))[:2 * limit]
        single_seeds = [] if cyclic_only else sorted({(key,) for color in anchors for key in self.by_color[color]}, key=priority)[:2 * limit]
        queue, seen = deque(cycle_seeds + single_seeds), set(cycle_seeds + single_seeds)
        growth = [set(), set(), set()]
        if cyclic_only:
            for idx, left in enumerate(cycle_seeds):
                for right in cycle_seeds[:idx]:
                    if steps >= self.max_steps:
                        break
                    steps += 1
                    joined = tuple(sorted(set(left) | set(right)))
                    if (max(len(left), len(right)) < len(joined) <= self.max_cores
                            and scope(left) & scope(right) & active):
                        growth[2].add(joined)
        while not cyclic_only and queue and steps < self.max_steps:
            keys = queue.popleft()
            if len(keys) >= self.max_cores:
                continue
            colors = scope(keys)
            frontier = set().union(*(set(self.by_color[color]) for color in colors & active)) - set(keys)
            for key in sorted(frontier, key=lambda key: (-len(self.scopes[key] & colors), priority((key,)))):
                if steps >= self.max_steps:
                    break
                steps += 1
                extended = tuple(sorted((*keys, key)))
                if extended in seen:
                    continue
                seen.add(extended)
                queue.append(extended)
                # Incidence-graph cycle rank distinguishes trees and joined loops.
                cycles = sum(len(self.scopes[key]) for key in extended) - len(extended) - len(scope(extended)) + 1
                growth[min(cycles, 2)].add(extended)
        families.extend(growth)

        def proposals():
            if not cyclic_only:
                yield from zip_longest(*(sorted(family, key=priority) for family in families))
                return
            pools = [sorted(family, key=priority)[:4 * limit] for family in families]
            uses = {}
            while any(pools):
                row = []
                for pool in pools:
                    keys = min(pool, key=lambda keys: (sum(uses.get(key, 0) for key in keys) / len(keys), priority(keys))) if pool else None
                    if keys is not None:
                        pool.remove(keys)
                        for key in keys:
                            uses[key] = uses.get(key, 0) + 1
                    row.append(keys)
                yield row

        result, emitted = [], set()
        for row in proposals():
            for keys in row:
                if keys is None:
                    continue
                outputs = scope(keys) & eligible
                if cyclic_only:
                    outputs &= active
                ranked = sorted(outputs, key=rank)
                if compact_domains:
                    affordable, output_size = [], 1
                    for color in ranked:
                        if len(affordable) < self.max_open_colors and output_size * self.dimensions[color] <= self.max_size:
                            affordable.append(color)
                            output_size *= self.dimensions[color]
                    ranked = affordable
                output = tuple(sorted(ranked[:self.max_open_colors]))
                signature = keys, output
                if not output or signature in emitted or all(sizes[color] == 1 for color in output):
                    continue
                emitted.add(signature)
                if prod(self.dimensions[color] for color in output) > self.max_size:
                    continue
                candidate = SummaryClusterCandidate("cluster_" + sha1(repr(signature).encode()).hexdigest(), keys, output)
                cost_key = (signature, tuple(sizes[color] for color in sorted(scope(keys)))) if compact_domains else signature
                if cost_key not in self._costs:
                    if len(self._costs) >= self.max_steps:
                        self._costs.pop(next(iter(self._costs)))
                    inputs = ({key: SimpleNamespace(colors=self.cores[key].colors,
                                                   shape=tuple(sizes[color] for color in self.cores[key].colors))
                               for key in keys} if compact_domains else self.cores)
                    self._costs[cost_key] = estimate_candidate_cost(inputs, candidate)
                cost = self._costs[cost_key]
                if cost.work <= self.max_work and cost.max_size <= self.max_size:
                    result.append(candidate)
                    if len(result) == limit:
                        return tuple(result)
        return tuple(result)
