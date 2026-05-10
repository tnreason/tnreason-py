import pandas as pd
from demonstrations.sudoku.examples import standard_sudoku
from demonstrations.sudoku.examples import sakana_fish
from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence

from .closure_engine import AlphaSudokuClosureEngine
from .env import AlphaSudokuEnv
from .mcts import AlphaSudokuMCTS
from .policies import HeuristicPolicyValue, PolicyValue
from .validation import validate_assignment

CHALLENGE100_PATH = "/home/schuette/Desktop/AlphaSudoku/tnreason-py-version2(2)/tnreason-py-version2/demonstrations/sudoku/sudoku_bench/sample_data/challenge100.csv"

def load_challenge100_puzzle(puzzle_num: int):
    df = pd.read_csv(CHALLENGE100_PATH)
    row = df.iloc[puzzle_num]
    evidence = initial_board_into_evidence(row["initial_board"])
    true_solution = initial_board_into_evidence(row["solution"])
    meta = {
        "puzzle_num": puzzle_num,
        "puzzle_id": row.get("puzzle_id"),
        "title": row.get("title"),
        "author": row.get("author"),
    }
    return evidence, true_solution, meta


def _solve_with_mcts(
    canetwork_factory,
    puzzle_evidence=None,
    true_solution=None,
    challenge_puzzle_num=None,
    policy_value: PolicyValue | None = None,
    num: int = 3,
    max_steps: int = 200,
    simulations: int = 200,
):
    if puzzle_evidence is None:
        if challenge_puzzle_num is None:
            raise ValueError("solve_with_mcts requires either puzzle_evidence or challenge_puzzle_num.")
        puzzle_evidence, loaded_solution, meta = load_challenge100_puzzle(challenge_puzzle_num)
        if true_solution is None:
            true_solution = loaded_solution
    else:
        meta = None

    def build_canetwork(assignment):
        return canetwork_factory(num=num, startAssignment=assignment)

    engine = AlphaSudokuClosureEngine(num=num, canetwork_factory=build_canetwork)
    env = AlphaSudokuEnv(engine)
    mcts = AlphaSudokuMCTS(
        env=env,
        policy_value=policy_value or HeuristicPolicyValue(),
        simulations=simulations,
    )

    state = env.reset(puzzle_evidence)
    for step in range(max_steps):
        if state.done:
            break
        action = mcts.choose_action(state)
        state, reward, done = env.step(state, action)
        print(f"step={step + 1} action={action} reward={reward:.3f} assigned={state.assigned_count}")
        if done:
            break

    validation_network = canetwork_factory(num=num, startAssignment={})
    valid, scalar = validate_assignment(validation_network, state.evidence)
    comparison = None
    if true_solution is not None:
        correct_atoms = sum(
            1 for atom_key, value in state.evidence.items()
            if value == 1 and atom_key in true_solution
        )
        comparison = {
            "exact_match": state.evidence == true_solution,
            "correct_atoms": correct_atoms,
            "solution_atoms": len(true_solution),
        }

    return state, valid, scalar, engine.cache_info(), comparison, meta

def solve_puzzle(
    canetwork_factory,
    challenge_puzzle_num: int = 3,
    policy_value: PolicyValue | None = None,
    num: int = 3,
    max_steps: int = 200,
    simulations: int = 200,
):
    policy_name = type(policy_value).__name__ if policy_value is not None else HeuristicPolicyValue.__name__
    print(
        f"Solving puzzle {challenge_puzzle_num} with MCTS "
        f"(policy={policy_name}, num={num}, max_steps={max_steps}, simulations={simulations})..."
    )
    return _solve_with_mcts(
        canetwork_factory=canetwork_factory,
        challenge_puzzle_num=challenge_puzzle_num,
        policy_value=policy_value,
        num=num,
        max_steps=max_steps,
        simulations=simulations,
    )


if __name__ == "__main__":
    import time
    start_time = time.time()
    ###################### Example usage: Solve a standard Sudoku puzzle from the Challenge100 dataset ######################
    final_state, valid, scalar, cache_info, comparison, meta = solve_puzzle(standard_sudoku.get_assignment_as_CANetwork, 3)
    # ###################### Example usage: Solve the Parity Fish puzzle from the Sakana Fish dataset ######################
    # final_state, valid, scalar, cache_info, comparison, meta = solve_puzzle(sakana_fish.get_assignment_as_CANetwork, 23)
    final_time = time.time()
    print(f"Solved in {final_time - start_time:.2f} seconds.")

    if meta is not None:
        print("Puzzle:", meta["puzzle_num"], meta["title"], "-", meta["author"])
    print("Solved:", final_state.solved)
    print("Contradiction:", final_state.contradiction)
    print("Assigned count:", final_state.assigned_count)
    print("Valid assignment:", valid, "score:", scalar)
    if comparison is not None:
        print("Matches known solution:", comparison["exact_match"])
        print("Correct atoms vs known solution:", comparison["correct_atoms"], "/", comparison["solution_atoms"])
    print("Closure cache:", cache_info)
