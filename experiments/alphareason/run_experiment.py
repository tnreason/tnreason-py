from demonstrations.sudoku.examples import standard_sudoku
from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence
import pandas as pd

from .closure_engine import AlphaReasonClosureEngine
from .env import AlphaReasonEnv
from .mcts import AlphaReasonMCTS
from .policies import HeuristicPolicyValue, PolicyValue
from .tasks.sudoku import build_standard_sudoku_task

CHALLENGE100_PATH = "/home/schuette/Desktop/AlphaSudoku/tnreason-py-version2(2)/tnreason-py-version2/demonstrations/sudoku/sudoku_bench/sample_data/challenge100.csv"


def load_challenge100_task(puzzle_num: int, num: int = 3):
    df = pd.read_csv(CHALLENGE100_PATH)
    row = df.iloc[puzzle_num]
    initial_evidence = initial_board_into_evidence(row["initial_board"])
    solution_evidence = initial_board_into_evidence(row["solution"])
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


if __name__ == "__main__":
    task = build_demo_sudoku_task()
    final_state, history, cache_info = solve_task(task, simulations=10, max_steps=10)
    print("Task:", task.name)
    print("Solved:", final_state.solved)
    print("Contradiction:", final_state.contradiction)
    print("Assigned target features:", final_state.assigned_count, "/", len(task.target_feature_keys))
    print("History length:", len(history))
    print("Closure cache:", cache_info)
