"""Structural treewidth and shape-only contraction planning; masks do not shrink axes."""
from dataclasses import dataclass
from collections import Counter
from heapq import heapify, heappop, heappush
from itertools import combinations
from math import prod
from numbers import Integral


@dataclass(frozen=True)
class ContractionEstimate:
    work: int
    max_size: int
    output_size: int
    path: tuple[tuple[int, ...], ...]


def estimate_treewidth(core_dict, candidate):
    """Min-fill upper bound on the selected cores' primal-graph treewidth.

    Variables sharing a core form a clique. This is unweighted graph width,
    not contraction width: dimensions, masks and chosen outputs do not affect it.
    Scalar-only clusters have width zero. No tensors are contracted or modified.
    """
    from networkx import Graph
    from networkx.algorithms.approximation import treewidth_min_fill_in

    if not candidate.core_keys or len(set(candidate.core_keys)) != len(candidate.core_keys):
        raise ValueError("Candidate needs nonempty, distinct core keys")
    scopes = [set(core_dict[key].colors) for key in candidate.core_keys]
    graph = Graph()
    graph.add_nodes_from(sorted(set().union(*scopes)))
    for scope in scopes:
        graph.add_edges_from(combinations(sorted(scope), 2))
    return max(0, treewidth_min_fill_in(graph)[0])


def estimate_candidate_cost(core_dict, candidate):
    """Estimate one deterministic greedy path, without allocating tensors.

    ``work`` sums the products of union-scope dimensions at each contraction;
    it is an approximate scalar-work count, not wall time or a FLOP count.
    ``max_size`` includes inputs and outputs, not total simultaneously-live
    storage. The greedy path minimizes the next intermediate, then work;
    it is not an optimal-path guarantee. Execute it with NumPy's
    ``optimize=['einsum_path', *estimate.path]`` to use the estimated order.
    Shared colors are retained until no remaining tensor needs them.
    """
    keys, output = tuple(candidate.core_keys), tuple(candidate.open_colors)
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Candidate needs nonempty, distinct core keys")
    if len(output) != len(set(output)):
        raise ValueError("Candidate open colors must be distinct")

    dims, scopes = {}, []
    for key in keys:
        core = core_dict[key]
        colors, shape = tuple(core.colors), tuple(core.shape)
        if len(colors) != len(shape) or len(colors) != len(set(colors)):
            raise ValueError(f"Invalid colors/shape for core {key!r}")
        for color, dim in zip(colors, shape):
            if isinstance(dim, bool) or not isinstance(dim, Integral) or dim <= 0:
                raise ValueError(f"Invalid dimension for color {color!r}")
            if color in dims and dims[color] != dim:
                raise ValueError(f"Inconsistent dimension for color {color!r}")
            dims[color] = int(dim)
        scopes.append(frozenset(colors))
    if not set(output) <= dims.keys():
        raise ValueError("Candidate open colors must occur in its selected cores")

    def size(scope):
        return prod(dims[color] for color in scope)

    open_set = frozenset(output)
    output_size = size(output)
    max_size, work, path = max(map(size, scopes)), 0, []
    if len(scopes) == 1:
        # A unary path also handles reduction/reordering of a single input.
        return ContractionEstimate(size(scopes[0]), max_size, output_size, ((0,),))
    scopes = dict(enumerate(scopes))
    occurrences = Counter(color for scope in scopes.values() for color in scope)

    def choice(left, right):
        a, b = scopes[left], scopes[right]
        union = a | b
        retained = frozenset(color for color in union if color in open_set or
                             occurrences[color] > (color in a) + (color in b))
        return size(retained), size(union), left, right, retained

    heap = [choice(left, right) for left, right in combinations(scopes, 2)]
    heapify(heap)
    fresh = len(scopes)
    while len(scopes) > 1:
        while True:
            intermediate_size, step_work, left, right, retained = heappop(heap)
            if left in scopes and right in scopes:
                break
        # Monotone tensor IDs preserve the original list's order and tie breaks.
        positions = {key: idx for idx, key in enumerate(scopes)}
        path.append((positions[left], positions[right]))
        for key in (left, right):
            occurrences.subtract(scopes.pop(key))
        scopes[fresh] = retained
        occurrences.update(retained)
        work += step_work
        max_size = max(max_size, intermediate_size)
        # Outside merges preserve whether two surviving scopes have another
        # owner of a shared color. Their scores remain valid in the heap.
        for key in scopes:
            if key != fresh:
                heappush(heap, choice(key, fresh))
        fresh += 1
    return ContractionEstimate(work, max(max_size, output_size), output_size, tuple(path))
