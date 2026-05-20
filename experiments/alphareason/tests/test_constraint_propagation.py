from tnreason import engine, representation

from ..constraint_propagation import (
    add_domain_cores,
    add_singleton_domain_cores,
    add_cluster_summary_core,
    enforce_generalized_arc_consistency,
)


def _equal_core(name, left, right):
    return engine.create_from_slice_iterator(
        colors=[left, right],
        shape=[2, 2],
        sliceIterator=[
            (1, {left: 0, right: 0}),
            (1, {left: 1, right: 1}),
        ],
        name=name,
    )


def test_generalized_arc_consistency_propagates_domain_reductions():
    core_dict = {
        "xy": _equal_core("xy", "x", "y"),
        "yz": _equal_core("yz", "y", "z"),
        "z_is_1": representation.create_basis_core(
            name="z_is_1",
            shape=[2],
            colors=["z"],
            numberTuple=[1],
        ),
    }

    domains, contradiction = enforce_generalized_arc_consistency(core_dict)

    assert not contradiction
    assert domains["x"] == (1,)
    assert domains["y"] == (1,)
    assert domains["z"] == (1,)


def test_generalized_arc_consistency_detects_empty_domain():
    core_dict = {
        "xy": _equal_core("xy", "x", "y"),
        "x_is_0": representation.create_basis_core(
            name="x_is_0",
            shape=[2],
            colors=["x"],
            numberTuple=[0],
        ),
        "y_is_1": representation.create_basis_core(
            name="y_is_1",
            shape=[2],
            colors=["y"],
            numberTuple=[1],
        ),
    }

    _, contradiction = enforce_generalized_arc_consistency(core_dict)

    assert contradiction


def test_add_singleton_domain_cores_adds_basis_cores():
    core_dict = {"xy": _equal_core("xy", "x", "y")}
    updated = add_singleton_domain_cores(core_dict, {"x": (1,), "y": (0, 1)})

    assert "gac_x" in updated
    assert updated["gac_x"][{"x": 1}] == 1
    assert updated["gac_x"][{"x": 0}] == 0
    assert "gac_y" not in updated


def test_add_domain_cores_adds_non_singleton_support_core():
    core_dict = {
        "x_domain": engine.create_from_slice_iterator(
            colors=["x"],
            shape=[3],
            sliceIterator=[
                (1, {"x": 0}),
                (1, {"x": 1}),
                (1, {"x": 2}),
            ],
        )
    }

    updated = add_domain_cores(core_dict, {"x": (0, 2)})

    assert "domain_x" in updated
    assert updated["domain_x"][{"x": 0}] == 1
    assert updated["domain_x"][{"x": 1}] == 0
    assert updated["domain_x"][{"x": 2}] == 1


def test_add_cluster_summary_core_keeps_requested_colors():
    core_dict = {
        "xy": _equal_core("xy", "x", "y"),
        "yz": _equal_core("yz", "y", "z"),
    }

    updated = add_cluster_summary_core(core_dict, open_colors=("x", "z"), name="xz_summary")

    assert "xz_summary" in updated
    assert updated["xz_summary"].colors == ["x", "z"]
    assert updated["xz_summary"][{"x": 0, "z": 0}] > 0
    assert updated["xz_summary"][{"x": 0, "z": 1}] == 0
