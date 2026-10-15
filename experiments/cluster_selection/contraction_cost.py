"""Shape-only contraction planning; zero masks do not reduce axis sizes."""
from dataclasses import dataclass
from itertools import combinations
from math import prod
from numbers import Integral


@dataclass(frozen=True)
class ContractionEstimate:
    work: int
    max_size: int
    output_size: int
    path: tuple[tuple[int, ...], ...]


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
    def choice(pair):
        union = scopes[pair[0]] | scopes[pair[1]]
        retained = union & open_set.union(*(scope for idx, scope in enumerate(scopes) if idx not in pair))
        return size(retained), size(union), *pair, retained

    while len(scopes) > 1:
        intermediate_size, step_work, left, right, retained = min(map(choice, combinations(range(len(scopes)), 2)), key=lambda c: c[:4])
        path.append((left, right))
        work += step_work
        max_size = max(max_size, intermediate_size)
        scopes = [scope for idx, scope in enumerate(scopes) if idx not in (left, right)]
        scopes.append(retained)
    return ContractionEstimate(work, max(max_size, output_size), output_size, tuple(path))
