from pathlib import Path
from dataclasses import dataclass
from typing import List
import warnings

from demonstrations.sudoku.examples import standard_sudoku
from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence
import pandas as pd

from .actions import AlphaReasonAction
from .closure_engine import AlphaReasonClosureEngine
from .env import AlphaReasonEnv
from .mcts import AlphaReasonMCTS
from .policies import HeuristicPolicyValue, PolicyValue
from .tasks.sudoku import (
    build_constraint_sudoku_task,
    build_sakana_fish_constraint_sudoku_task,
    build_standard_sudoku_task,
)
from .tasks.sudoku_examples import load_standard_example_evidence

REPO_ROOT = Path(__file__).resolve().parents[2]
CHALLENGE100_PATH = REPO_ROOT / "demonstrations" / "sudoku" / "sudoku_bench" / "sample_data" / "challenge100.csv"
SAKANA_FISH_CHALLENGE100_INDEX = 23


@dataclass
class BacktrackingFrame:
    state: object
    actions: List[AlphaReasonAction]
    next_action_index: int = 0
    incoming_action: AlphaReasonAction | None = None


@dataclass
class BacktrackingDiagnostics:
    stop_reason: str
    attempts: int
    max_depth: int
    remaining_stack_depth: int
    best_assigned_count: int


def _format_action(action: AlphaReasonAction) -> str:
    if hasattr(action, "feature_key") and hasattr(action, "value_index"):
        return f"{action.feature_key}={action.value_index}"
    if hasattr(action, "cluster_key"):
        return f"cluster:{action.cluster_key}"
    return repr(action)


def count_correct_target_assignments(task, state, solution_assignments=None) -> tuple[int, int]:
    solution_assignments = solution_assignments if solution_assignments is not None else task.solution_assignments
    if not solution_assignments:
        return 0, 0
    correct = sum(
        1
        for feature_key, value_index in state.target_assignments.items()
        if solution_assignments.get(feature_key) == value_index
    )
    return correct, len(state.target_assignments)


def solution_assignments_from_evidence(solution_evidence, target_feature_keys) -> dict[str, int]:
    solution_assignments = {}
    for feature_key in target_feature_keys:
        atom_prefix = "a_" + feature_key.removeprefix("pos_") + "_"
        for atom_key, atom_value in solution_evidence.items():
            if atom_value == 1 and atom_key.startswith(atom_prefix):
                solution_assignments[feature_key] = int(atom_key.removeprefix(atom_prefix))
                break
    return solution_assignments


def load_challenge100_task(puzzle_num: int, num: int = 3):
    df = pd.read_csv(CHALLENGE100_PATH)
    row = df.iloc[puzzle_num]
    initial_evidence = initial_board_into_evidence(str(row["initial_board"]))
    solution_evidence = initial_board_into_evidence(str(row["solution"]))
    task = build_standard_sudoku_task(
        initial_evidence=initial_evidence,
        solution_evidence=solution_evidence,
        num=num,
        name=f"standard_sudoku_{puzzle_num}",
    )
    meta = {
        "puzzle_num": puzzle_num,
        "title": row.get("title"),
        "author": row.get("author"),
    }
    return task, meta


def solve_task(
    task,
    policy_value: PolicyValue | None = None,
    max_steps: int = 200,
    simulations: int = 200,
    rollout_depth: int = 0,
):
    engine = AlphaReasonClosureEngine()
    env = AlphaReasonEnv(task=task, closure_engine=engine)
    mcts = AlphaReasonMCTS(
        env=env,
        policy_value=policy_value or HeuristicPolicyValue(),
        simulations=simulations,
        rollout_depth=rollout_depth,
    )

    state = env.reset()
    history = []
    for step in range(max_steps):
        if state.done:
            break
        action = mcts.choose_action(state)
        state, reward, done = env.step(state, action)
        history.append((action, reward, state.assigned_count))
        print(
            f"MCTS chose {_format_action(action)} -> reward={reward:.4f} "
            f"assigned={state.assigned_count}/{len(task.target_feature_keys)} "
            f"contradiction={state.contradiction}"
        )
        if done:
            break

    return state, history, engine.cache_info()


def solve_task_with_backtracking(
    task,
    policy_value: PolicyValue | None = None,
    max_steps: int = 200,
    simulations: int = 200,
    rollout_depth: int = 0,
    return_diagnostics: bool = False,
):
    engine = AlphaReasonClosureEngine()
    env = AlphaReasonEnv(task=task, closure_engine=engine)
    mcts = AlphaReasonMCTS(
        env=env,
        policy_value=policy_value or HeuristicPolicyValue(),
        simulations=simulations,
        rollout_depth=rollout_depth,
    )

    root_state = env.reset()
    history = []
    if root_state.done:
        return root_state, history, engine.cache_info()

    final_state = root_state
    best_state = root_state
    attempts = 0
    max_depth_reached = 0
    stop_reason = "running"

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in multiply",
            category=RuntimeWarning,
        )
        stack = [BacktrackingFrame(state=root_state, actions=mcts.rank_actions(root_state))]
        while stack and attempts < max_steps:
            max_depth_reached = max(max_depth_reached, len(stack))
            frame = stack[-1]
            final_state = frame.state

            if frame.state.solved:
                stop_reason = "solved"
                break

            if frame.next_action_index >= len(frame.actions):
                if frame.incoming_action is not None:
                    print(
                        f"Backtracking from depth={len(stack)} after exhausting "
                        f"{_format_action(frame.incoming_action)}"
                    )
                else:
                    print("Backtracking from root after exhausting all actions")
                stack.pop()
                continue

            action = frame.actions[frame.next_action_index]
            frame.next_action_index += 1
            next_state, reward, _ = env.step(frame.state, action)
            attempts += 1
            history.append((action, reward, next_state.assigned_count))
            final_state = next_state
            print(
                f"MCTS chose {_format_action(action)} at depth={len(stack)} -> reward={reward:.4f} "
                f"assigned={next_state.assigned_count}/{len(task.target_feature_keys)} "
                f"contradiction={next_state.contradiction}"
            )

            if next_state.contradiction:
                print(
                    f"Contradiction found after {_format_action(action)} "
                    f"at depth={len(stack)}"
                )
                continue
            if next_state.assigned_count > best_state.assigned_count:
                best_state = next_state
            if next_state.solved:
                final_state = next_state
                stop_reason = "solved"
                break

            stack.append(
                BacktrackingFrame(
                    state=next_state,
                    actions=mcts.rank_actions(next_state),
                    incoming_action=action,
                )
            )

    if stop_reason == "running":
        if attempts >= max_steps:
            stop_reason = "max_steps"
            final_state = stack[-1].state if stack else best_state
        elif not stack:
            stop_reason = "exhausted"
            final_state = best_state

    diagnostics = BacktrackingDiagnostics(
        stop_reason=stop_reason,
        attempts=attempts,
        max_depth=max_depth_reached,
        remaining_stack_depth=len(stack),
        best_assigned_count=best_state.assigned_count,
    )
    if return_diagnostics:
        return final_state, history, engine.cache_info(), diagnostics
    return final_state, history, engine.cache_info()


def build_demo_sudoku_task(
    initial: str = (
        "1..."
        ".4.."
        "..4."
        "...1"
    ),
    solution: str = (
        "1234"
        "3412"
        "2143"
        "4321"
    ),
    name: str = "demo_standard_sudoku_4x4",
):
    return build_standard_sudoku_task(
        initial_evidence=initial_board_into_evidence(initial, colNum=2, rowNum=2),
        solution_evidence=initial_board_into_evidence(solution, colNum=2, rowNum=2),
        num=2,
        name=name,
    )


def build_demo_constraint_sudoku_task(
    initial: str = (
        "1..."
        ".4.."
        "..4."
        "...1"
    ),
    solution: str = (
        "1234"
        "3412"
        "2143"
        "4321"
    ),
    name: str = "demo_constraint_sudoku_4x4",
):
    return build_constraint_sudoku_task(
        network_factory=lambda evidence: standard_sudoku.get_assignment_as_constraint_network(
            num=2,
            startAssignment=evidence,
        ),
        initial_evidence=initial_board_into_evidence(initial, colNum=2, rowNum=2),
        solution_evidence=initial_board_into_evidence(solution, colNum=2, rowNum=2),
        num=2,
        name=name,
    )


def build_n3_example_constraint_sudoku_task(name: str = "n3_example_constraint_sudoku"):
    evidence = load_standard_example_evidence("n=3")
    return build_constraint_sudoku_task(
        network_factory=lambda assignment: standard_sudoku.get_assignment_as_constraint_network(
            num=3,
            startAssignment=assignment,
        ),
        initial_evidence=evidence,
        solution_evidence=evidence,
        num=3,
        name=name,
    )


def run_sakana_fish_constraint_sudoku(
    max_steps: int = 5,
    simulations: int = 5,
    rollout_depth: int = 5,
):
    task = build_sakana_fish_constraint_sudoku_task()
    return solve_task_with_backtracking(
        task,
        max_steps=max_steps,
        simulations=simulations,
        rollout_depth=rollout_depth,
    )


if __name__ == "__main__":
    import time
    task = build_sakana_fish_constraint_sudoku_task()
    # task = build_n3_example_constraint_sudoku_task()
    start = time.time()
    final_state, history, cache_info, diagnostics = solve_task_with_backtracking(
        task,
        simulations=5000,
        max_steps=10000,
        rollout_depth=1000,
        return_diagnostics=True,
    )
    print(f"Solved in {time.time() - start:.2f} seconds")
    print("Task:", task.name)
    print("Solved:", final_state.solved)
    print("Contradiction:", final_state.contradiction)
    print("Assigned target features:", final_state.assigned_count, "/", len(task.target_feature_keys))
    row = pd.read_csv(CHALLENGE100_PATH).iloc[SAKANA_FISH_CHALLENGE100_INDEX]
    solution_evidence = initial_board_into_evidence(str(row["solution"]))
    solution_assignments = solution_assignments_from_evidence(
        solution_evidence,
        task.target_feature_keys,
    )
    correct_count, compared_count = count_correct_target_assignments(
        task,
        final_state,
        solution_assignments=solution_assignments,
    )
    print("Correct assigned target features:", correct_count, "/", compared_count)
    print("History length:", len(history))
    print("Closure cache:", cache_info)
    print("Stop reason:", diagnostics.stop_reason)
    print("Attempts:", diagnostics.attempts)
    print("Max depth:", diagnostics.max_depth)
    print("Backtracking:", diagnostics)
