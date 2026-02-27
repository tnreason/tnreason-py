import pandas as pd
from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence

from .closure_engine import AlphaSudokuClosureEngine
from .env import AlphaSudokuEnv
from .mcts import AlphaSudokuMCTS
from .validation import validate_sudoku_assignment


def solve_with_mcts(puzzle_num: int = 1, max_steps: int = 200, simulations: int = 200):
    path = "/home/schuette/Desktop/AlphaSudoku/tnreason-py-version2(2)/tnreason-py-version2/demonstrations/sudoku/sudoku_bench/sample_data/nikoli100.parquet"
    df = pd.read_parquet(path)
    evidence = initial_board_into_evidence(df["initial_board"][puzzle_num])
    true_solution = initial_board_into_evidence(df["solution"][puzzle_num])

    engine = AlphaSudokuClosureEngine(num=3)
    env = AlphaSudokuEnv(engine)
    mcts = AlphaSudokuMCTS(env=env, simulations=simulations)

    state = env.reset(evidence)
    for step in range(max_steps):
        if state.done:
            break
        action = mcts.choose_action(state)
        state, reward, done = env.step(state, action)
        print(f"step={step + 1} action={action} reward={reward:.3f} assigned={state.assigned_count}")
        if done:
            break

    valid, scalar = validate_sudoku_assignment(state.evidence, num=3)
    correct_atoms = sum(
        1 for atom_key, value in state.evidence.items()
        if value == 1 and atom_key in true_solution
    )
    exact_match = state.evidence == true_solution
    return state, valid, scalar, engine.cache_info(), {
        "exact_match": exact_match,
        "correct_atoms": correct_atoms,
        "solution_atoms": len(true_solution),
    }


if __name__ == "__main__":
    final_state, valid, scalar, cache_info, comparison = solve_with_mcts()
    print("Solved:", final_state.solved)
    print("Contradiction:", final_state.contradiction)
    print("Assigned count:", final_state.assigned_count)
    print("Valid assignment:", valid, "score:", scalar)
    print("Matches known solution:", comparison["exact_match"])
    print("Correct atoms vs known solution:", comparison["correct_atoms"], "/", comparison["solution_atoms"])
    print("Closure cache:", cache_info)
