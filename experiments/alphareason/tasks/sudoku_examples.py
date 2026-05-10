import ast
from pathlib import Path
from typing import Callable, Dict

from demonstrations.sudoku.examples import sakana_fish, standard_sudoku
from experiments.alphasudoku.validation import check_assignment_consistency

from ..task import AlphaReasonTask
from .sudoku import (
    _assignment_evidence_map,
    _solution_assignments_from_atom_evidence,
    _target_feature_keys,
    canetwork_closure,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "demonstrations" / "sudoku" / "examples"


def _literal_assignment_from_source(path: Path, variable_name: str):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == variable_name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Could not find literal assignment for {variable_name} in {path}.")


def load_standard_example_evidence(example_name: str) -> Dict[str, int]:
    if example_name not in {"n=2", "n=3"}:
        raise ValueError(f"Unsupported standard Sudoku example {example_name!r}.")
    return dict(_literal_assignment_from_source(EXAMPLES_DIR / f"{example_name}.py", "evidenceDict"))


def build_sudoku_example_task(
    example_name: str,
    *,
    initial_evidence: Dict[str, int] | None = None,
    solution_evidence: Dict[str, int] | None = None,
) -> AlphaReasonTask:
    if example_name == "n=2":
        num = 2
        evidence = initial_evidence or load_standard_example_evidence(example_name)
        network_factory: Callable[[Dict[str, int]], object] = (
            lambda assignment: standard_sudoku.get_assignment_as_CANetwork(num=num, startAssignment=assignment)
        )
    elif example_name == "n=3":
        num = 3
        evidence = initial_evidence or load_standard_example_evidence(example_name)
        network_factory = lambda assignment: standard_sudoku.get_assignment_as_CANetwork(
            num=num,
            startAssignment=assignment,
        )
    elif example_name == "sakana_fish":
        num = 3
        evidence = dict(initial_evidence or {})
        network_factory = lambda assignment: sakana_fish.get_assignment_as_CANetwork(
            num=num,
            startAssignment=assignment,
        )
    else:
        raise ValueError(f"Unsupported Sudoku example {example_name!r}.")

    target_feature_keys = _target_feature_keys(num)
    solution = solution_evidence or evidence
    return AlphaReasonTask(
        name=f"sudoku_example_{example_name}",
        network_factory=network_factory,
        closure_function=canetwork_closure,
        initial_evidence=dict(evidence),
        target_feature_keys=target_feature_keys,
        feature_domain_sizes={feature_key: num ** 2 for feature_key in target_feature_keys},
        assignment_evidence_map=_assignment_evidence_map(num),
        solution_assignments=_solution_assignments_from_atom_evidence(solution, num=num),
        consistency_checker=check_assignment_consistency,
        metadata={"num": num, "example_name": example_name},
    )
