from demonstrations.sudoku.examples import standard_sudoku
from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence

from .dfs_solver import AlphaReasonDFSSolver
from .closure_engine import AlphaReasonClosureEngine
from .tasks.sudoku import (
    build_constraint_sudoku_task,
    build_prepared_standard_sudoku_task,
    build_sakana_fish_constraint_sudoku_task,
)


def assignments_from_evidence(task, evidence):
    assignments = {}
    for feature_key in task.target_feature_keys:
        for value_index, value_evidence in task.assignment_evidence_map[feature_key].items():
            if any(evidence.get(key) == value for key, value in value_evidence.items()):
                assignments[feature_key] = value_index
                break
    return assignments


def print_grid(task, assignments) -> None:
    num = task.metadata["num"]
    side = num ** 2
    for row in range(side):
        values = []
        for col in range(side):
            r1, r2 = divmod(row, num)
            c1, c2 = divmod(col, num)
            feature_key = f"pos_{r1}_{r2}_{c1}_{c2}"
            value = assignments.get(feature_key)
            values.append("." if value is None else str(value + 1))
        print(" ".join(values))


def board_string(task, assignments) -> str:
    num = task.metadata["num"]
    side = num ** 2
    values = []
    for row in range(side):
        for col in range(side):
            r1, r2 = divmod(row, num)
            c1, c2 = divmod(col, num)
            feature_key = f"pos_{r1}_{r2}_{c1}_{c2}"
            value = assignments.get(feature_key)
            values.append("." if value is None else str(value + 1))
    return "".join(values)


def compare_to_solution(task, state):
    if not task.solution_assignments:
        print("Reference solution: unavailable")
        return

    mismatches = [
        (feature_key, state.target_assignments.get(feature_key), solution_value)
        for feature_key, solution_value in task.solution_assignments.items()
        if state.target_assignments.get(feature_key) != solution_value
    ]
    print("Reference solution available:", True)
    print("Matches reference:", not mismatches)
    print("Correct cells:", len(task.solution_assignments) - len(mismatches), "/", len(task.solution_assignments))
    print("Final solution string:", board_string(task, state.target_assignments))
    print("Reference solution string:", board_string(task, task.solution_assignments))
    if mismatches:
        print("Mismatches:")
        for feature_key, found, expected in mismatches[:20]:
            found_text = "." if found is None else str(found + 1)
            print(f"  {feature_key}: found={found_text}, expected={expected + 1}")


if __name__ == "__main__":
    # for ind in range(3000):
    #     task = build_prepared_standard_sudoku_task(row_index=ind)

    # task = build_constraint_sudoku_task(
    #     network_factory=lambda evidence: standard_sudoku.get_assignment_as_constraint_network(
    #         num=2,
    #         startAssignment=evidence,
    #     ),
    #     initial_evidence=initial_board_into_evidence(
    #         "1..."
    #         ".4.."
    #         "..4."
    #         "...1",
    #         colNum=2,
    #         rowNum=2,
    #     ),
    #     solution_evidence=initial_board_into_evidence(
    #         "1234"
    #         "3412"
    #         "2143"
    #         "4321",
    #         colNum=2,
    #         rowNum=2,
    #     ),
    #     num=2,
    #     name="basic_standard_sudoku_4x4",
    # )

    task = build_sakana_fish_constraint_sudoku_task()

    closure_engine = AlphaReasonClosureEngine()
    initial_state = closure_engine.close(task, task.initial_evidence, active_inference_clusters={})
    solver = AlphaReasonDFSSolver(
        closure_engine=closure_engine,
        max_nodes=100,
        branch_feature_count=1,
        trace=True,
    )

    # print(ind)
    print("Task:", task.name)
    print("Source:", task.metadata.get("source"))
    print("Puzzle id:", task.metadata.get("puzzle_id"))
    print("Prepared assigned:", task.metadata.get("prepared_assigned_count"))
    print()
    print("Initial grid:")
    print_grid(task, assignments_from_evidence(task, task.initial_evidence))
    print()
    print("After initial forward chaining closure:")
    print_grid(task, initial_state.target_assignments)
    if task.solution_assignments:
        print()
        print("Loaded reference solution:")
        print_grid(task, task.solution_assignments)
    print()
    print("Search:")
    result = solver.solve(task)
    print()
    print("Solved:", result.solved)
    print("Contradiction:", result.state.contradiction)
    print("Assigned:", result.state.assigned_count, "/", len(task.target_feature_keys))
    print("Nodes:", result.nodes)
    print("History length:", len(result.history))
    compare_to_solution(task, result.state)
    print()
    print("Final grid:")
    print_grid(task, result.state.target_assignments)
    if task.solution_assignments:
        print()
        print("Reference grid:")
        print_grid(task, task.solution_assignments)
