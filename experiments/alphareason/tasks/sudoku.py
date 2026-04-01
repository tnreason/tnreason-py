import demonstrations.sudoku.inferenceClusters as inf_clusters
from demonstrations.sudoku.examples import standard_sudoku
from experiments.alphasudoku.validation import check_assignment_consistency

from ..task import AlphaReasonTask


def _cell_feature_key(num: int, row: int, col: int) -> str:
    r1 = row // num
    r2 = row % num
    c1 = col // num
    c2 = col % num
    return f"pos_{r1}_{r2}_{c1}_{c2}"


def _cell_atom_key(num: int, row: int, col: int, digit: int) -> str:
    r1 = row // num
    r2 = row % num
    c1 = col // num
    c2 = col % num
    return f"a_{r1}_{r2}_{c1}_{c2}_{digit}"


def _solution_assignments_from_atom_evidence(solution_evidence, num: int):
    assignments = {}
    for row in range(num ** 2):
        for col in range(num ** 2):
            feature_key = _cell_feature_key(num, row, col)
            for digit in range(num ** 2):
                atom_key = _cell_atom_key(num, row, col, digit)
                if solution_evidence.get(atom_key) == 1:
                    assignments[feature_key] = digit
                    break
    return assignments


def build_standard_sudoku_task(
    initial_evidence,
    solution_evidence,
    num: int = 3,
    name: str = "standard_sudoku",
    include_rule_detectors: bool = True,
):
    target_feature_keys = tuple(
        _cell_feature_key(num, row, col)
        for row in range(num ** 2)
        for col in range(num ** 2)
    )
    feature_domain_sizes = {feature_key: num ** 2 for feature_key in target_feature_keys}
    assignment_evidence_map = {
        feature_key: {
            digit: {_cell_atom_key(num, row, col, digit): 1}
            for digit in range(num ** 2)
        }
        for row in range(num ** 2)
        for col in range(num ** 2)
        for feature_key in [_cell_feature_key(num, row, col)]
    }

    rule_detectors = ()
    if include_rule_detectors:
        rule_detectors = (
            inf_clusters.detect_locked_candidates_square,
            inf_clusters.detect_locked_candidates_row,
            inf_clusters.detect_locked_candidates_col,
            inf_clusters.detect_hidden_pairs,
            inf_clusters.detect_naked_pairs,
        )

    return AlphaReasonTask(
        name=name,
        canetwork_factory=lambda evidence: standard_sudoku.get_assignment_as_CANetwork(num=num, startAssignment=evidence),
        initial_evidence=dict(initial_evidence),
        target_feature_keys=target_feature_keys,
        feature_domain_sizes=feature_domain_sizes,
        assignment_evidence_map=assignment_evidence_map,
        solution_assignments=_solution_assignments_from_atom_evidence(solution_evidence, num=num),
        rule_detectors=rule_detectors,
        rule_detector_kwargs={"num": num},
        consistency_checker=check_assignment_consistency,
        metadata={"num": num},
    )

