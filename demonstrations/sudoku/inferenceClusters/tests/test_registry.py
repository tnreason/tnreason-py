import importlib

import pytest

import demonstrations.sudoku.inferenceClusters as rules
from demonstrations.sudoku.inferenceClusters.core import RuleContext, RuleLimits
from demonstrations.sudoku.inferenceClusters.registry import (
    RuleRegistry, UnsupportedRuleError, available_rules, detect_rules,
)
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.sudoku_tests.standard_constraints import get_sudoku_constraint_network


def context():
    net = ConstraintNetwork(get_sudoku_constraint_network(num=2))
    return RuleContext(net, num=2, limits=RuleLimits(max_steps=1))


def test_registry_requires_explicit_finite_selection_and_rejects_unencoded_rules():
    assert RuleRegistry([]).detect(context()) == {}
    for names in (("unique_rectangles",), ("bug",), ("variant_rules",), ("unknown_rule",)):
        with pytest.raises(UnsupportedRuleError):
            RuleRegistry(names)
    with pytest.raises(TypeError, match="finite"):
        RuleRegistry(iter(("singles",)))
    with pytest.raises(TypeError):
        detect_rules(context())


def test_only_selected_detectors_execute_with_independent_budgets():
    selected = ("naked_subsets", "xy_wing")
    result = detect_rules(context(), selected, options={"naked_subsets": {"sizes": (2,)}})
    assert tuple(result) == selected
    assert all(item.steps == 1 and item.truncated for item in result.values())
    with pytest.raises(ValueError, match="unselected"):
        detect_rules(context(), ("singles",), options={"basic_fish": {"sizes": (2,)}})


def test_import_does_not_detect_or_modify_generic_mac(monkeypatch):
    monkeypatch.setattr(RuleContext, "scan", lambda self: pytest.fail("Detection ran on import"))
    importlib.reload(rules)
    net = ConstraintNetwork(get_sudoku_constraint_network(num=2))
    state = net.create_mac_state({"pos_0_0_0_0": (0, 1), "pos_0_0_1_0": (0, 1)})
    keys = set(state._cores)
    assert state.on_closure is None and state.propagate()
    assert set(state._cores) == keys and state.domains["pos_0_0_0_1"] == (0, 1, 2, 3)


@pytest.mark.parametrize("name", available_rules())
def test_every_explicit_selector_respects_a_tiny_scan_budget(name):
    detection = detect_rules(context(), (name,))[name]
    assert detection.steps <= 1
    assert not detection.clusters


def test_family_options_cannot_override_a_fixed_registry_selection():
    with pytest.raises(TypeError):
        detect_rules(context(), ("naked_subsets",), options={"naked_subsets": {"hidden": True}})
    with pytest.raises(TypeError):
        detect_rules(context(), ("basic_fish",), options={"basic_fish": {"kind": "finned"}})
    with pytest.raises(TypeError):
        detect_rules(context(), ("aligned_pair_exclusion",),
                     options={"aligned_pair_exclusion": {"size": 3}})
