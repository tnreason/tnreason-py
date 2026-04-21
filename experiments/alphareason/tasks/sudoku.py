import demonstrations.sudoku.inferenceClusters as inf_clusters
from demonstrations.sudoku.examples import sakana_fish, standard_sudoku
from experiments.alphasudoku.validation import check_assignment_consistency
from experiments.constraint_networks.forward_chaining import check_local_satisfiability

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


def _target_feature_keys(num: int):
    return tuple(
        _cell_feature_key(num, row, col)
        for row in range(num ** 2)
        for col in range(num ** 2)
    )


def _assignment_evidence_map(num: int):
    return {
        feature_key: {
            digit: {_cell_atom_key(num, row, col, digit): 1}
            for digit in range(num ** 2)
        }
        for row in range(num ** 2)
        for col in range(num ** 2)
        for feature_key in [_cell_feature_key(num, row, col)]
    }


def canetwork_closure(closure_engine, task, evidence, active_inference_clusters):
    return closure_engine.close_canetwork(task, evidence, active_inference_clusters)


def constraint_network_closure(closure_engine, task, evidence, active_inference_clusters):
    return closure_engine.close_constraint_network(task, evidence, active_inference_clusters)


def build_standard_sudoku_task(
    initial_evidence,
    solution_evidence,
    num: int = 3,
    name: str = "standard_sudoku",
    include_rule_detectors: bool = True,
):
    target_feature_keys = _target_feature_keys(num)
    feature_domain_sizes = {feature_key: num ** 2 for feature_key in target_feature_keys}

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
        network_factory=lambda evidence: standard_sudoku.get_assignment_as_CANetwork(num=num, startAssignment=evidence),
        closure_function=canetwork_closure,
        initial_evidence=dict(initial_evidence),
        target_feature_keys=target_feature_keys,
        feature_domain_sizes=feature_domain_sizes,
        assignment_evidence_map=_assignment_evidence_map(num),
        solution_assignments=_solution_assignments_from_atom_evidence(solution_evidence, num=num),
        rule_detectors=rule_detectors,
        rule_detector_kwargs={"num": num},
        consistency_checker=check_assignment_consistency,
        metadata={"num": num},
    )


def build_constraint_sudoku_task(
    network_factory,
    initial_evidence,
    solution_evidence,
    num: int = 3,
    name: str = "constraint_sudoku",
):
    target_feature_keys = _target_feature_keys(num)
    feature_domain_sizes = {feature_key: num ** 2 for feature_key in target_feature_keys}

    return AlphaReasonTask(
        name=name,
        network_factory=network_factory,
        closure_function=constraint_network_closure,
        initial_evidence=dict(initial_evidence),
        target_feature_keys=target_feature_keys,
        feature_domain_sizes=feature_domain_sizes,
        assignment_evidence_map=_assignment_evidence_map(num),
        solution_assignments=_solution_assignments_from_atom_evidence(solution_evidence, num=num),
        consistency_checker=lambda constraint_network, assignment: check_local_satisfiability(constraint_network.coresDict),
        metadata={"num": num},
    )


def build_sakana_fish_constraint_sudoku_task(
    initial_evidence=None,
    solution_evidence=None,
    num: int = 3,
    name: str = "sakana_fish_constraint_sudoku",
):
    initial_evidence = dict(initial_evidence or {})
    solution_evidence = dict(solution_evidence or initial_evidence)
    return build_constraint_sudoku_task(
        network_factory=lambda evidence: sakana_fish.get_assignment_as_constraint_network(
            num=num,
            startAssignment=evidence,
        ),
        initial_evidence=initial_evidence,
        solution_evidence=solution_evidence,
        num=num,
        name=name,
    )
