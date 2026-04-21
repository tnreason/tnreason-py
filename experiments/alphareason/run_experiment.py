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


@dataclass
class BacktrackingFrame:
    state: object
    actions: List[AlphaReasonAction]
    next_action_index: int = 0


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


def solve_task(task, policy_value: PolicyValue | None = None, max_steps: int = 200, simulations: int = 200):
    engine = AlphaReasonClosureEngine()
    env = AlphaReasonEnv(task=task, closure_engine=engine)
    mcts = AlphaReasonMCTS(env=env, policy_value=policy_value or HeuristicPolicyValue(), simulations=simulations)

    state = env.reset()
    history = []
    for step in range(max_steps):
        if state.done:
            break
        action = mcts.choose_action(state)
        state, reward, done = env.step(state, action)
        history.append((action, reward, state.assigned_count))
        if done:
            break

    return state, history, engine.cache_info()


def solve_task_with_backtracking(
    task,
    policy_value: PolicyValue | None = None,
    max_steps: int = 200,
    simulations: int = 200,
):
    engine = AlphaReasonClosureEngine()
    env = AlphaReasonEnv(task=task, closure_engine=engine)
    mcts = AlphaReasonMCTS(env=env, policy_value=policy_value or HeuristicPolicyValue(), simulations=simulations)

    root_state = env.reset()
    history = []
    if root_state.done:
        return root_state, history, engine.cache_info()

    final_state = root_state
    best_state = root_state
    attempts = 0

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in multiply",
            category=RuntimeWarning,
        )
        stack = [BacktrackingFrame(state=root_state, actions=mcts.rank_actions(root_state))]
        while stack and attempts < max_steps:
            frame = stack[-1]
            final_state = frame.state

            if frame.state.solved:
                return frame.state, history, engine.cache_info()

            if frame.next_action_index >= len(frame.actions):
                stack.pop()
                continue

            action = frame.actions[frame.next_action_index]
            frame.next_action_index += 1
            next_state, reward, _ = env.step(frame.state, action)
            attempts += 1
            history.append((action, reward, next_state.assigned_count))
            final_state = next_state

            if next_state.contradiction:
                continue
            if next_state.assigned_count > best_state.assigned_count:
                best_state = next_state
            if next_state.solved:
                return next_state, history, engine.cache_info()

            stack.append(BacktrackingFrame(state=next_state, actions=mcts.rank_actions(next_state)))

    if stack:
        return stack[-1].state, history, engine.cache_info()
    return best_state, history, engine.cache_info()


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


def run_sakana_fish_constraint_sudoku(max_steps: int = 5, simulations: int = 5):
    task = build_sakana_fish_constraint_sudoku_task()
    return solve_task_with_backtracking(task, max_steps=max_steps, simulations=simulations)


if __name__ == "__main__":
    task = build_sakana_fish_constraint_sudoku_task()
    final_state, history, cache_info = solve_task_with_backtracking(task, simulations=50, max_steps=200)
    print("Task:", task.name)
    print("Solved:", final_state.solved)
    print("Contradiction:", final_state.contradiction)
    print("Assigned target features:", final_state.assigned_count, "/", len(task.target_feature_keys))
    print("History length:", len(history))
    print("Closure cache:", cache_info)
