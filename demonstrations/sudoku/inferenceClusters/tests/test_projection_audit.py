"""Independent regression checks for compaction and original proof interfaces."""
from itertools import product

import numpy as np
import pytest

from demonstrations.sudoku.inferenceClusters.core import RuleContext, contract_cluster
from demonstrations.sudoku.inferenceClusters.projections import (
    describe_pattern_overlay, describe_templates, detect_pattern_overlay,
)
from experiments.constraint_networks.sudoku_tests.standard_constraints import get_sudoku_constraint_network
from tnreason.engine.workload_to_numpy import NumpyCore


@pytest.mark.parametrize("fixed", ["none", "mixed", "all"])
@pytest.mark.parametrize("scalar", [0, -3])
@pytest.mark.parametrize("use_domains", [False, True])
def test_high_degree_originals_disconnected_scalars_and_fixed_joint_outputs(fixed, scalar, use_domains):
    rng = np.random.default_rng(431)
    specs = [
        (("a_probe", "probe_x"), (2, 3)),
        (("probe_y", "a_probe"), (2, 2)),
        (("a_probe", "probe_fixed"), (2, 2)),
        (("probe_x", "a_probe", "probe_y"), (3, 2, 2)),
        (("probe_y", "probe_x"), (2, 3)),
    ]
    original = get_sudoku_constraint_network(num=1)
    keys = []
    for index, (colors, shape) in enumerate(specs):
        key = f"audit_factor_{index}"
        original[key] = NumpyCore(values=rng.choice((-2, 0, 3), size=shape, p=(.4, .2, .4)),
                                  colors=list(colors))
        keys.append(key)
    extras = {
        "audit_scalar": NumpyCore(values=np.asarray(scalar), colors=[]),
        "audit_detached": NumpyCore(values=np.array([0, -2, 3]), colors=["probe_detached"]),
        "audit_fixed": NumpyCore(values=np.array([-2, 3]), colors=["probe_fixed"]),
    }
    original.update(extras)
    keys.extend(extras)
    outputs = ("probe_fixed", "a_probe", "probe_y")
    domains = {"probe_x": (0, 2), "probe_detached": (1, 2)}
    if fixed in ("mixed", "all"):
        domains["probe_fixed"] = (1,)
    if fixed == "all":
        domains.update({"a_probe": (1,), "probe_y": (0,)})
    ctx = RuleContext(original, domains, num=1)
    cluster = ctx.describe_tensors("audit", tuple(keys), outputs)
    preserved = {key: core.values.copy() for key, core in original.items()}
    expected = np.zeros((2, 2, 2), dtype=bool)
    colors = cluster.participating_variables
    possibilities = [ctx.domains[color] if use_domains else range(ctx.dimensions[color]) for color in colors]
    for values in product(*possibilities):
        assignment = dict(zip(colors, values))
        if all(ctx.cores[key][{color: assignment[color] for color in ctx.cores[key].colors}] != 0
               for key in keys):
            expected[tuple(assignment[color] for color in outputs)] = True
    joint = contract_cluster(ctx, cluster, joint=True, use_domains=use_domains)
    unary = contract_cluster(ctx, cluster, use_domains=use_domains)
    assert joint.status == unary.status == "executed"
    np.testing.assert_array_equal(next(iter(joint.summaries.values())).values, expected)
    assert joint.contradiction == unary.contradiction == (not expected.any())
    for axis, color in enumerate(outputs):
        values = np.any(expected, axis=tuple(i for i in range(3) if i != axis))
        assert unary.supported_domains[color] == tuple(np.flatnonzero(values))
    for key, core in original.items():
        np.testing.assert_array_equal(core.values, preserved[key])


@pytest.mark.parametrize("selector", [
    lambda ctx, outputs: describe_templates(ctx, 0, outputs=outputs),
    lambda ctx, outputs: describe_pattern_overlay(ctx, (0, 1), outputs=outputs),
    lambda ctx, outputs: detect_pattern_overlay(ctx, digits=(0, 1), outputs=outputs),
])
def test_public_projection_outputs_require_finite_explicit_collections(selector):
    ctx = RuleContext(get_sudoku_constraint_network(num=2), num=2)
    # An ordinary finite iterator establishes rejection without hanging a
    # regression run if the check is accidentally removed for infinite inputs.
    with pytest.raises(TypeError):
        selector(ctx, iter((ctx.unit("row", 0, 0),)))
