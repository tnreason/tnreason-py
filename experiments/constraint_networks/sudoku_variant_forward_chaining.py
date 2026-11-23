from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Tuple

from experiments.alphareason.closure_engine import AlphaReasonClosureEngine
from experiments.constraint_networks.generalized_arc_consistency import add_domain_cores


Cell = Tuple[int, int]
Domains = Dict[str, set[int]]
BinaryPredicate = Callable[[int, int], bool]
BinaryConstraint = Tuple[str, str, BinaryPredicate]
TableConstraint = Tuple[Tuple[str, ...], frozenset[Tuple[int, ...]]]


def cell_key(row: int, col: int, num: int = 3) -> str:
    r1, r2 = divmod(row, num)
    c1, c2 = divmod(col, num)
    return f"pos_{r1}_{r2}_{c1}_{c2}"


def parse_cell_key(feature_key: str, num: int = 3) -> Cell | None:
    parts = feature_key.split("_")
    if len(parts) != 5 or parts[0] != "pos":
        return None
    try:
        r1, r2, c1, c2 = (int(part) for part in parts[1:])
    except ValueError:
        return None
    if not all(0 <= part < num for part in (r1, r2, c1, c2)):
        return None
    return r1 * num + r2, c1 * num + c2


def parse_atom_key(atom_key: str, num: int = 3) -> Tuple[str, int] | None:
    parts = atom_key.split("_")
    if len(parts) != 6 or parts[0] != "a":
        return None
    feature_key = f"pos_{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}"
    if parse_cell_key(feature_key, num=num) is None:
        return None
    try:
        digit = int(parts[5])
    except ValueError:
        return None
    if not 0 <= digit < num**2:
        return None
    return feature_key, digit


def all_cell_keys(num: int = 3) -> Tuple[str, ...]:
    side = num**2
    return tuple(cell_key(row, col, num=num) for row in range(side) for col in range(side))


def sudoku_units(num: int = 3) -> Tuple[Tuple[str, ...], ...]:
    side = num**2
    units = []
    for row in range(side):
        units.append(tuple(cell_key(row, col, num=num) for col in range(side)))
    for col in range(side):
        units.append(tuple(cell_key(row, col, num=num) for row in range(side)))
    for block_row in range(num):
        for block_col in range(num):
            units.append(
                tuple(
                    cell_key(block_row * num + r, block_col * num + c, num=num)
                    for r in range(num)
                    for c in range(num)
                )
            )
    return tuple(units)


def edge_keys(edges: Iterable[Tuple[Cell, Cell]], num: int = 3) -> Tuple[Tuple[str, str], ...]:
    return tuple((cell_key(*first, num=num), cell_key(*second, num=num)) for first, second in edges)


def white_dot_allowed(left: int, right: int) -> bool:
    return abs(left - right) == 1


def black_dot_allowed(left: int, right: int) -> bool:
    left_digit = left + 1
    right_digit = right + 1
    return left_digit == 2 * right_digit or right_digit == 2 * left_digit


def red_line_allowed(left: int, right: int) -> bool:
    return left % 2 != right % 2


@dataclass
class SudokuVariantPropagationResult:
    domains: Domains
    contradiction: bool
    iterations: int
    reason: str | None = None


def _domain_text(domains: Mapping[str, Iterable[int]], feature_key: str) -> str:
    return f"{feature_key}={tuple(value + 1 for value in sorted(domains[feature_key]))}"


class SudokuVariantForwardChainer:
    def __init__(
        self,
        num: int = 3,
        white_dot_edges: Iterable[Tuple[Cell, Cell]] = (),
        black_dot_edges: Iterable[Tuple[Cell, Cell]] = (),
        parity_edges: Iterable[Tuple[Cell, Cell]] = (),
        binary_constraints: Iterable[Tuple[Cell, Cell, BinaryPredicate]] = (),
        table_constraints: Iterable[TableConstraint] = (),
    ):
        self.num = num
        self.side = num**2
        self.cell_keys = all_cell_keys(num=num)
        self.units = sudoku_units(num=num)
        self.binary_constraints: Tuple[BinaryConstraint, ...] = (
            *((left, right, white_dot_allowed) for left, right in edge_keys(white_dot_edges, num=num)),
            *((left, right, black_dot_allowed) for left, right in edge_keys(black_dot_edges, num=num)),
            *((left, right, red_line_allowed) for left, right in edge_keys(parity_edges, num=num)),
            *(
                (cell_key(*left, num=num), cell_key(*right, num=num), allowed)
                for left, right, allowed in binary_constraints
            ),
        )
        self.table_constraints: Tuple[TableConstraint, ...] = tuple(table_constraints)

    def initial_domains(self, evidence: Mapping[str, int]) -> Domains:
        domains = {feature_key: set(range(self.side)) for feature_key in self.cell_keys}
        for key, value in evidence.items():
            if value != 1:
                continue
            parsed_atom = parse_atom_key(key, num=self.num)
            if parsed_atom is not None:
                feature_key, digit = parsed_atom
                if feature_key in domains:
                    domains[feature_key] = {digit}
                continue
            if key in domains:
                domains[key] = {int(value)}
        for key, value in evidence.items():
            if key in domains and value != 1:
                domains[key] = {int(value)}
        return domains

    def propagate(self, evidence: Mapping[str, int]) -> SudokuVariantPropagationResult:
        domains = self.initial_domains(evidence)
        contradiction = False
        iterations = 0
        changed = True
        while changed and not contradiction:
            changed = False
            iterations += 1

            unit_changed, contradiction, reason = self.propagate_units(domains)
            changed = changed or unit_changed
            if contradiction:
                return SudokuVariantPropagationResult(domains, True, iterations, reason)

            edge_changed, contradiction, reason = self.propagate_binary_constraints(domains)
            changed = changed or edge_changed
            if contradiction:
                return SudokuVariantPropagationResult(domains, True, iterations, reason)

            table_changed, contradiction, reason = self.propagate_table_constraints(domains)
            changed = changed or table_changed
            if contradiction:
                return SudokuVariantPropagationResult(domains, True, iterations, reason)

        empty_domains = [feature_key for feature_key, domain in domains.items() if not domain]
        if empty_domains:
            reason = "empty domains after propagation: " + ", ".join(empty_domains[:5])
            return SudokuVariantPropagationResult(domains, True, iterations, reason)
        return SudokuVariantPropagationResult(domains, False, iterations)

    def propagate_units(self, domains: Domains) -> Tuple[bool, bool, str | None]:
        changed = False
        for unit in self.units:
            assigned = {}
            for feature_key in unit:
                if len(domains[feature_key]) == 1:
                    digit = next(iter(domains[feature_key]))
                    if digit in assigned:
                        reason = (
                            "unit conflict: "
                            f"{feature_key} and {assigned[digit]} are both {digit + 1}"
                        )
                        return changed, True, reason
                    assigned[digit] = feature_key

            for digit, assigned_key in assigned.items():
                for feature_key in unit:
                    if feature_key == assigned_key:
                        continue
                    if digit in domains[feature_key]:
                        domains[feature_key].remove(digit)
                        changed = True
                        if not domains[feature_key]:
                            reason = (
                                "unit propagation emptied domain: "
                                f"removed {digit + 1} from {feature_key} because of {assigned_key}"
                            )
                            return changed, True, reason

            for digit in range(self.side):
                places = [feature_key for feature_key in unit if digit in domains[feature_key]]
                if not places:
                    reason = (
                        "unit conflict: "
                        f"digit {digit + 1} has no possible position in unit {unit}"
                    )
                    return changed, True, reason
                if len(places) == 1 and len(domains[places[0]]) > 1:
                    domains[places[0]] = {digit}
                    changed = True
        return changed, False, None

    def propagate_binary_constraints(self, domains: Domains) -> Tuple[bool, bool, str | None]:
        changed = False
        for left, right, allowed in self.binary_constraints:
            left_support = {
                left_digit
                for left_digit in domains[left]
                if any(allowed(left_digit, right_digit) for right_digit in domains[right])
            }
            right_support = {
                right_digit
                for right_digit in domains[right]
                if any(allowed(left_digit, right_digit) for left_digit in domains[left])
            }
            if not left_support or not right_support:
                reason = (
                    "binary constraint has no support: "
                    f"{_domain_text(domains, left)} with {_domain_text(domains, right)}"
                )
                return changed, True, reason
            if left_support != domains[left]:
                domains[left] = left_support
                changed = True
            if right_support != domains[right]:
                domains[right] = right_support
                changed = True
        return changed, False, None

    def propagate_table_constraints(self, domains: Domains) -> Tuple[bool, bool, str | None]:
        changed = False
        for colors, allowed_tuples in self.table_constraints:
            supported_values = {color: set() for color in colors}
            found_supported_tuple = False
            current_domains = tuple(domains[color] for color in colors)

            for allowed_tuple in allowed_tuples:
                if not all(
                    value in domain
                    for value, domain in zip(allowed_tuple, current_domains)
                ):
                    continue
                found_supported_tuple = True
                for color, value in zip(colors, allowed_tuple):
                    supported_values[color].add(value)

            if not found_supported_tuple:
                reason = (
                    "table constraint has no supported tuple: "
                    f"colors={colors}, "
                    f"domains={tuple(_domain_text(domains, color) for color in colors)}, "
                    f"allowed_tuples={len(allowed_tuples)}"
                )
                return changed, True, reason

            for color in colors:
                restricted_domain = domains[color] & supported_values[color]
                if not restricted_domain:
                    reason = (
                        "table constraint emptied domain: "
                        f"{color}, colors={colors}, allowed_tuples={len(allowed_tuples)}"
                    )
                    return changed, True, reason
                if restricted_domain != domains[color]:
                    domains[color] = restricted_domain
                    changed = True
        return changed, False, None


def _build_state_from_domains(
    closure_engine,
    task,
    evidence,
    active_inference_clusters,
    domains: Mapping[str, Iterable[int]],
    contradiction: bool,
    message_count: int,
):
    normalized_domains = {feature_key: set(domain) for feature_key, domain in domains.items()}
    closed_evidence = dict(evidence)
    target_assignments = {}
    feature_supports = {}
    feature_priors = {}
    unresolved_features = []

    for feature_key in task.available_action_feature_keys():
        domain_size = task.feature_domain_sizes[feature_key]
        supported = sorted(normalized_domains.get(feature_key, set(range(domain_size))))
        feature_supports[feature_key] = tuple(supported)
        feature_priors[feature_key] = closure_engine._distribution_from_supported(domain_size, supported)
        if len(supported) == 1:
            value_index = supported[0]
            if feature_key in task.target_feature_keys:
                target_assignments[feature_key] = value_index
            closed_evidence.update(task.assignment_to_evidence(feature_key, value_index))
        elif len(supported) == 0:
            contradiction = True
        else:
            unresolved_features.append((feature_key, supported))

    consistency_ok = not contradiction
    return closure_engine._build_state(
        task=task,
        evidence=closed_evidence,
        active_inference_clusters=active_inference_clusters,
        target_assignments=target_assignments,
        feature_supports=feature_supports,
        feature_priors=feature_priors,
        unresolved_features=unresolved_features,
        contradiction=contradiction,
        consistency_ok=consistency_ok,
        propagator=None,
        message_count=message_count,
    )


def _reduced_domains(domains: Mapping[str, Iterable[int]], feature_domain_sizes: Mapping[str, int]):
    reduced = {}
    for feature_key, domain in domains.items():
        if feature_key not in feature_domain_sizes:
            continue
        domain_tuple = tuple(sorted(set(domain)))
        if domain_tuple != tuple(range(feature_domain_sizes[feature_key])):
            reduced[feature_key] = domain_tuple
    return reduced


class SudokuVariantHybridForwardChainingClosureEngine(AlphaReasonClosureEngine):
    def __init__(self, num: int = 3, binary_domain_prefix: str = "sudoku_binary", **kwargs):
        super().__init__(**kwargs)
        self.num = num
        self.binary_domain_prefix = binary_domain_prefix
        self._binary_chainer_cache = {}
        self.last_contradiction_reason = None

    def _binary_chainer(self, core_dict):
        return ConstraintNetworkSudokuVariantForwardChainer(core_dict, num=self.num)

    def _cached_binary_chainer(self, task):
        cache_key = task.name
        if cache_key not in self._binary_chainer_cache:
            self._binary_chainer_cache[cache_key] = self._binary_chainer(
                task.network_factory({})
            )
        return self._binary_chainer_cache[cache_key]

    def close_constraint_network(self, task, evidence, active_inference_clusters):
        if active_inference_clusters:
            core_dict = task.network_factory(evidence)
            core_dict = self._prepare_constraint_network(core_dict, active_inference_clusters)
            chainer = self._binary_chainer(core_dict)
        else:
            core_dict = None
            chainer = self._cached_binary_chainer(task)
        result = chainer.propagate(evidence)
        self.last_contradiction_reason = result.reason
        if result.contradiction:
            return _build_state_from_domains(
                closure_engine=self,
                task=task,
                evidence=evidence,
                active_inference_clusters=active_inference_clusters,
                domains=result.domains,
                contradiction=True,
                message_count=result.iterations,
            )

        if chainer.fully_supported and not active_inference_clusters:
            return _build_state_from_domains(
                closure_engine=self,
                task=task,
                evidence=evidence,
                active_inference_clusters=active_inference_clusters,
                domains=result.domains,
                contradiction=False,
                message_count=result.iterations,
            )

        if core_dict is None:
            core_dict = task.network_factory(evidence)
        core_dict = add_domain_cores(
            core_dict,
            _reduced_domains(result.domains, task.feature_domain_sizes),
            prefix=self.binary_domain_prefix,
        )
        return self._close_prepared_constraint_network(
            task=task,
            evidence=evidence,
            active_inference_clusters=active_inference_clusters,
            core_dict=core_dict,
        )


def _support_pairs(core, left_color: str, right_color: str) -> frozenset[Tuple[int, int]]:
    left_axis = list(core.colors).index(left_color)
    right_axis = list(core.colors).index(right_color)
    pairs = set()
    values = getattr(core, "values", None)
    if values is not None:
        import numpy as np

        for index in np.argwhere(np.asarray(values) != 0):
            pairs.add((int(index[left_axis]), int(index[right_axis])))
        return frozenset(pairs)

    for value, assignment in core:
        if value != 0:
            pairs.add((int(assignment[left_color]), int(assignment[right_color])))
    return frozenset(pairs)


def _support_tuples(core, colors: Tuple[str, ...]) -> frozenset[Tuple[int, ...]]:
    axes = tuple(list(core.colors).index(color) for color in colors)
    tuples = set()
    values = getattr(core, "values", None)
    if values is not None:
        import numpy as np

        for index in np.argwhere(np.asarray(values) != 0):
            tuples.add(tuple(int(index[axis]) for axis in axes))
        return frozenset(tuples)

    for value, assignment in core:
        if value != 0:
            tuples.add(tuple(int(assignment[color]) for color in colors))
    return frozenset(tuples)


def _table_allowed(allowed_pairs: frozenset[Tuple[int, int]]) -> BinaryPredicate:
    return lambda left, right, pairs=allowed_pairs: (left, right) in pairs


class ConstraintNetworkSudokuVariantForwardChainer(SudokuVariantForwardChainer):
    def __init__(self, core_dict: Mapping[str, object], num: int = 3):
        self.core_dict = core_dict
        self.num = num
        self.unsupported_core_keys = []
        binary_constraints, table_constraints = self._compile_constraints(core_dict)
        super().__init__(
            num=num,
            binary_constraints=binary_constraints,
            table_constraints=table_constraints,
        )

    @property
    def fully_supported(self) -> bool:
        return not self.unsupported_core_keys

    def _compile_constraints(
        self,
        core_dict: Mapping[str, object],
    ) -> Tuple[Tuple[Tuple[Cell, Cell, BinaryPredicate], ...], Tuple[TableConstraint, ...]]:
        constraints = []
        table_constraints = []
        odd_indicator_links: Dict[str, Tuple[str, frozenset[Tuple[int, int]]]] = {}
        odd_indicator_edges = []

        for core_key, core in core_dict.items():
            colors = list(getattr(core, "colors", ()))
            if self._is_standard_sudoku_core(colors):
                continue
            if len(colors) == 1:
                pos_color = colors[0]
                if parse_cell_key(pos_color, num=self.num) is not None:
                    table_constraints.append(
                        ((pos_color,), _support_tuples(core, (pos_color,)))
                    )
                continue

            pos_colors = [color for color in colors if parse_cell_key(color, num=self.num) is not None]
            odd_colors = [color for color in colors if color.startswith("oddInd_")]

            if len(pos_colors) == 2:
                left, right = pos_colors
                constraints.append(
                    (
                        parse_cell_key(left, num=self.num),
                        parse_cell_key(right, num=self.num),
                        _table_allowed(_support_pairs(core, left, right)),
                    )
                )
            elif len(pos_colors) == 1 and len(odd_colors) == 1:
                pos_color = pos_colors[0]
                odd_color = odd_colors[0]
                odd_indicator_links[odd_color] = (
                    pos_color,
                    _support_pairs(core, pos_color, odd_color),
                )
            elif len(odd_colors) == 2:
                left_odd, right_odd = odd_colors
                odd_indicator_edges.append(
                    (core_key, left_odd, right_odd, _support_pairs(core, left_odd, right_odd))
                )
            elif len(pos_colors) == len(colors):
                table_colors = tuple(pos_colors)
                table_constraints.append(
                    (table_colors, _support_tuples(core, table_colors))
                )
            else:
                self.unsupported_core_keys.append(core_key)

        constraints.extend(
            self._compile_odd_indicator_constraints(odd_indicator_links, odd_indicator_edges)
        )
        return (
            tuple(
                constraint
                for constraint in constraints
                if constraint[0] is not None and constraint[1] is not None
            ),
            tuple(table_constraints),
        )

    def _is_standard_sudoku_core(self, colors: Iterable[str]) -> bool:
        colors = tuple(colors)
        if len(colors) != 2:
            return False
        left, right = colors
        if left.startswith("a_") and (
            right.startswith(("row_", "col_", "square_")) or parse_cell_key(right, num=self.num) is not None
        ):
            return True
        return right.startswith("a_") and (
            left.startswith(("row_", "col_", "square_")) or parse_cell_key(left, num=self.num) is not None
        )

    def _compile_odd_indicator_constraints(
        self,
        odd_indicator_links: Mapping[str, Tuple[str, frozenset[Tuple[int, int]]]],
        odd_indicator_edges: Iterable[Tuple[str, str, str, frozenset[Tuple[int, int]]]],
    ) -> Tuple[Tuple[Cell, Cell, BinaryPredicate], ...]:
        constraints = []
        for core_key, left_odd, right_odd, odd_pairs in odd_indicator_edges:
            if left_odd not in odd_indicator_links or right_odd not in odd_indicator_links:
                self.unsupported_core_keys.append(core_key)
                continue
            left_pos, left_pairs = odd_indicator_links[left_odd]
            right_pos, right_pairs = odd_indicator_links[right_odd]
            allowed_digit_pairs = frozenset(
                (left_digit, right_digit)
                for left_digit, left_odd_value in left_pairs
                for right_digit, right_odd_value in right_pairs
                if (left_odd_value, right_odd_value) in odd_pairs
            )
            constraints.append(
                (
                    parse_cell_key(left_pos, num=self.num),
                    parse_cell_key(right_pos, num=self.num),
                    _table_allowed(allowed_digit_pairs),
                )
            )
        return tuple(constraints)
