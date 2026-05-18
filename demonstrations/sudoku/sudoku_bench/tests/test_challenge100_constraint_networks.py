import importlib

import pytest


EXAMPLE_MODULES = (
    "zeeta_moth",
    "parity_fish",
    "ascension",
    "four_triangles",
    "astraeus",
    "tendril_battle",
    "corners",
    "zip_zag",
    "willow",
    "slow_thermos",
    "four_corner_shots",
    "mutation",
    "arbol",
    "fibonacci_lines",
    "challenge100_43",
    "broken_paperclip",
    "the_two_headed_martian",
    "reticule",
    "raw_ping",
    "what_s_the_difference",
    "transmission",
    "n_is_for_naughty_and_nice",
    "nostalgia",
    "northwest_wind",
    "uraninite",
    "pumpkin_wizard",
    "stronghold",
    "introduction_to_killer_sudoku",
    "introduction_to_renban_lines",
    "introduction_to_anti_knight",
    "pluto_dwarf_planet",
    "knights_of_the_qs_example",
    "leftovers",
    "nautilus",
    "mc_hammer",
)


def test_supported_indices_are_explicit_square_subset():
    assert "zeeta_moth" in EXAMPLE_MODULES
    assert "willow" in EXAMPLE_MODULES
    assert "introduction_to_killer_sudoku" in EXAMPLE_MODULES
    assert "challenge100_45" not in EXAMPLE_MODULES


def test_zeeta_moth_variant_cores_include_all_clue_families():
    module = importlib.import_module("demonstrations.sudoku.examples.zeeta_moth")
    cores = module.get_zeeta_moth_constraint_cores()
    names = set(cores)

    assert any("cage" in name for name in names)
    assert any("_v_" in name for name in names)
    assert any("_x_" in name for name in names)
    assert any("thermo" in name for name in names)
    assert any("renban" in name for name in names)
    assert any("entropic" in name for name in names)


def test_parity_fish_variant_cores_match_sakana_style_rules():
    module = importlib.import_module("demonstrations.sudoku.examples.parity_fish")
    cores = module.get_parity_fish_constraint_cores()
    names = set(cores)

    assert any("white_dot" in name for name in names)
    assert any("black_dot" in name for name in names)
    assert any(name.startswith("red_line_") for name in names)


def test_blank_killer_cages_become_no_repeat_constraints():
    module = importlib.import_module("demonstrations.sudoku.examples.willow")
    cores = module.get_willow_constraint_cores()

    assert any("no_repeat_cage" in name for name in cores)


def test_full_network_can_include_standard_cores_and_evidence():
    killer = importlib.import_module(
        "demonstrations.sudoku.examples.introduction_to_killer_sudoku"
    )
    network = killer.get_assignment_as_constraint_network()

    assert "pos_0_0_0_0_a_0_0_0_0_0" in network
    assert any("little_killer" in name for name in network)

    renban = importlib.import_module(
        "demonstrations.sudoku.examples.introduction_to_renban_lines"
    )
    network_with_given = renban.get_assignment_as_constraint_network()
    assert any(name.startswith("a_") for name in network_with_given)


def test_unsupported_rows_have_no_example_module():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("demonstrations.sudoku.examples.challenge100_45")


def test_named_example_modules_expose_constraint_network_builders():
    for module_name in EXAMPLE_MODULES:
        module = importlib.import_module(
            f"demonstrations.sudoku.examples.{module_name}"
        )
        specific_core_builders = [
            value
            for name, value in vars(module).items()
            if name.startswith("get_")
            and name.endswith("_constraint_cores")
            and getattr(value, "__module__", None) == module.__name__
        ]

        assert len(specific_core_builders) == 1
        assert isinstance(specific_core_builders[0](), dict)
        assert isinstance(
            module.get_assignment_as_constraint_network(include_initial_board=False),
            dict,
        )
