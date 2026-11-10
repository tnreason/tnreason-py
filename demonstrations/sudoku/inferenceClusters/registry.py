"""Caller-selected rule dispatch; constructing/importing never runs detection."""
from .advanced import (
    detect_als_xz, detect_simple_coloring, detect_xy_chains, detect_xy_wings,
    detect_xyz_wings,
)
from .subsets_fish import detect_fish, detect_locked, detect_singles, detect_subsets
from .fish_general import (detect_general_fish, detect_franken_fish, detect_mutant_fish,
                           detect_multi_fin_fish, detect_endo_fish,
                           detect_cannibalistic_fish, detect_siamese_fish)
from .implication import (detect_aic, detect_x_chains, detect_grouped_aic,
                          detect_nice_loops, detect_turbot_fish, detect_skyscrapers,
                          detect_two_string_kites, detect_empty_rectangles,
                          detect_w_wings, detect_multi_coloring, detect_medusa)
from .als_extended import (detect_als_xz_extended, detect_als_chains, detect_als_xy_wing,
                           detect_death_blossom, detect_bent_subsets, detect_wxyz_wings,
                           detect_sue_de_coq, detect_aligned_exclusion)
from .projections import (detect_templates, detect_pattern_overlay, detect_nishio,
                          detect_proof_regions, detect_complete_support)
from .forcing import (detect_forcing_chains, detect_forcing_nets,
                       detect_kraken_fish, detect_kraken_type2)


class UnsupportedRuleError(ValueError):
    pass


# These are function definitions for explicit dispatch, not solver registration.
_DETECTORS = {
    "singles": detect_singles,
    "locked_candidates": detect_locked,
    "naked_subsets": lambda ctx, **opts: detect_subsets(ctx, hidden=False, **opts),
    "hidden_subsets": lambda ctx, **opts: detect_subsets(ctx, hidden=True, **opts),
    "basic_fish": lambda ctx, **opts: detect_fish(ctx, kind="basic", **opts),
    "finned_fish": lambda ctx, **opts: detect_fish(ctx, kind="finned", **opts),
    "sashimi_fish": lambda ctx, **opts: detect_fish(ctx, kind="sashimi", **opts),
    "xy_wing": detect_xy_wings,
    "xyz_wing": detect_xyz_wings,
    "simple_coloring": detect_simple_coloring,
    "xy_chain": detect_xy_chains,
    "als_xz": detect_als_xz,
    "general_fish": detect_general_fish,
    "franken_fish": detect_franken_fish,
    "mutant_fish": detect_mutant_fish,
    "multi_fin_fish": detect_multi_fin_fish,
    "endo_fish": detect_endo_fish,
    "cannibalistic_fish": detect_cannibalistic_fish,
    "siamese_fish": detect_siamese_fish,
    "aic": detect_aic,
    "x_chain": detect_x_chains,
    "grouped_aic": detect_grouped_aic,
    "nice_loops": detect_nice_loops,
    "grouped_nice_loops": lambda ctx, **opts: detect_nice_loops(ctx, grouped=True, **opts),
    "x_cycles": lambda ctx, **opts: detect_nice_loops(ctx, single_digit=True, **opts),
    "turbot_fish": detect_turbot_fish,
    "skyscraper": detect_skyscrapers,
    "two_string_kite": detect_two_string_kites,
    "empty_rectangle": detect_empty_rectangles,
    "w_wing": detect_w_wings,
    "multi_coloring": detect_multi_coloring,
    "medusa": detect_medusa,
    "als_xz_extended": detect_als_xz_extended,
    "doubly_linked_als": lambda ctx, **opts: detect_als_xz_extended(ctx, kind="doubly", **opts),
    "als_chains": detect_als_chains,
    "als_xy_wing": detect_als_xy_wing,
    "death_blossom": detect_death_blossom,
    "bent_subsets": detect_bent_subsets,
    "wxyz_wing": detect_wxyz_wings,
    "sue_de_coq": detect_sue_de_coq,
    "aligned_pair_exclusion": lambda ctx, **opts: detect_aligned_exclusion(ctx, size=2, **opts),
    "aligned_triple_exclusion": lambda ctx, **opts: detect_aligned_exclusion(ctx, size=3, **opts),
    "templates": detect_templates,
    "pattern_overlay": detect_pattern_overlay,
    "nishio": detect_nishio,
    "proof_regions": detect_proof_regions,
    "complete_support": detect_complete_support,
    "forcing_chains": detect_forcing_chains,
    "forcing_nets": detect_forcing_nets,
    "kraken_fish": detect_kraken_fish,
    "kraken_type2": detect_kraken_type2,
}

UNSUPPORTED_RULES = {
    "unique_rectangles": "Requires a unique-solution axiom absent from ordinary Sudoku constraints.",
    "bug": "Requires uniqueness and exact BUG incidence conditions; neither is encoded here.",
    "avoidable_rectangles": "Requires uniqueness and given/non-given provenance.",
    "variant_rules": "Requires variant constraints (diagonal, cages, anti-knight, etc.) in the original network.",
}


def available_rules():
    """Names accepted by explicit dispatch, without selecting or running any."""
    return tuple(_DETECTORS)


class RuleRegistry:
    """An explicit, finite family selection, independent of MAC or a policy.

    Each family has its own context limits. For F selected families, total
    detection is bounded by F * max_steps and F * max_patterns. Results retain
    per-family truncation information. Option dictionaries are passed only to
    their named family; unused option names are rejected.
    """
    def __init__(self, families):
        if not isinstance(families, (tuple, list, set, frozenset)):
            raise TypeError("families must be a finite collection of explicit rule names")
        if len(families) > len(_DETECTORS):
            raise ValueError("Too many selected rule names")
        if isinstance(families, (set, frozenset)):
            families = sorted(families)
        selected = tuple(dict.fromkeys(families))
        for name in selected:
            if name not in _DETECTORS:
                reason = UNSUPPORTED_RULES.get(name, "No implemented selector has this name.")
                raise UnsupportedRuleError(f"{name}: {reason}")
        self.families = selected

    def detect(self, context, *, options=None):
        options = dict(options or {})
        if options.keys() - set(self.families):
            raise ValueError("Options refer to an unselected rule family")
        return {name: _DETECTORS[name](context, **dict(options.get(name, {}))) for name in self.families}


def detect_rules(context, families, *, options=None):
    """Run only explicitly selected families; return name -> Detection."""
    return RuleRegistry(families).detect(context, options=options)
