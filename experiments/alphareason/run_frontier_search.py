from dataclasses import dataclass, field
import heapq
from itertools import count
from pathlib import Path
from time import time
from typing import List
import warnings

import pandas as pd

from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence

from .actions import AlphaReasonAction
from .closure_engine import AlphaReasonClosureEngine
from .env import AlphaReasonEnv
from .mcts import AlphaReasonMCTS
from .policies import HeuristicPolicyValue, PolicyValue
from .run_experiment import (
    SAKANA_FISH_CHALLENGE100_INDEX,
    count_correct_target_assignments,
    solution_assignments_from_evidence,
)
from .state import AlphaReasonState
from .tasks.sudoku import build_sakana_fish_constraint_sudoku_task


REPO_ROOT = Path(__file__).resolve().parents[2]
CHALLENGE100_PATH = REPO_ROOT / "demonstrations" / "sudoku" / "sudoku_bench" / "sample_data" / "challenge100.csv"


@dataclass(order=True)
class FrontierItem:
    priority: float
    order: int
    state: AlphaReasonState = field(compare=False)
    history: List[tuple[AlphaReasonAction, float, int]] = field(compare=False, default_factory=list)


@dataclass
class FrontierDiagnostics:
    stop_reason: str
    expansions: int
    generated_states: int
    contradiction_count: int
    max_frontier_size: int
    best_assigned_count: int


def _format_action(action: AlphaReasonAction) -> str:
    if hasattr(action, "feature_key") and hasattr(action, "value_index"):
        return f"{action.feature_key}={action.value_index}"
    if hasattr(action, "cluster_key"):
        return f"cluster:{action.cluster_key}"
    return repr(action)


def _frontier_score(state: AlphaReasonState, history_length: int) -> float:
    progress = state.assigned_count / max(1, len(state.target_feature_keys))
    return (
        3.0 * progress
        + state.value_estimate
        - 0.002 * history_length
    )


def solve_task_with_frontier_search(
    task,
    policy_value: PolicyValue | None = None,
    max_expansions: int = 1000,
    simulations: int = 100,
    rollout_depth: int = 0,
    expand_action_count: int = 5,
    frontier_size: int = 5000,
):
    engine = AlphaReasonClosureEngine()
    env = AlphaReasonEnv(task=task, closure_engine=engine)
    mcts = AlphaReasonMCTS(
        env=env,
        policy_value=policy_value,
        simulations=simulations,
        rollout_depth=rollout_depth,
    )

    root_state = env.reset()
    if root_state.done:
        diagnostics = FrontierDiagnostics(
            stop_reason="root_done",
            expansions=0,
            generated_states=1,
            contradiction_count=int(root_state.contradiction),
            max_frontier_size=0,
            best_assigned_count=root_state.assigned_count,
        )
        return root_state, [], engine.cache_info(), diagnostics

    queue_counter = count()
    frontier: list[FrontierItem] = []
    heapq.heappush(
        frontier,
        FrontierItem(
            priority=-_frontier_score(root_state, 0),
            order=next(queue_counter),
            state=root_state,
            history=[],
        ),
    )
    best_state = root_state
    best_history: List[tuple[AlphaReasonAction, float, int]] = []
    generated_states = 1
    contradiction_count = 0
    max_frontier_size = 1
    expansions = 0
    visited_evidence = {tuple(sorted(root_state.evidence.items()))}

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in multiply",
            category=RuntimeWarning,
        )
        while frontier and expansions < max_expansions:
            item = heapq.heappop(frontier)
            state = item.state
            history = item.history

            if state.solved:
                diagnostics = FrontierDiagnostics(
                    stop_reason="solved",
                    expansions=expansions,
                    generated_states=generated_states,
                    contradiction_count=contradiction_count,
                    max_frontier_size=max_frontier_size,
                    best_assigned_count=state.assigned_count,
                )
                return state, history, engine.cache_info(), diagnostics

            ranked_actions = mcts.rank_actions(state)[:expand_action_count]
            if not ranked_actions:
                continue

            expansions += 1
            for action in ranked_actions:
                next_state, reward, _ = env.step(state, action)
                generated_states += 1
                next_history = [*history, (action, reward, next_state.assigned_count)]
                print(
                    f"Frontier expanded {_format_action(action)} from depth={len(history)} "
                    f"-> reward={reward:.4f} assigned={next_state.assigned_count}/{len(task.target_feature_keys)} "
                    f"contradiction={next_state.contradiction}"
                )

                if next_state.contradiction:
                    contradiction_count += 1
                    print(f"Contradiction found after {_format_action(action)}")
                    continue

                if next_state.assigned_count > best_state.assigned_count:
                    best_state = next_state
                    best_history = next_history

                if next_state.solved:
                    diagnostics = FrontierDiagnostics(
                        stop_reason="solved",
                        expansions=expansions,
                        generated_states=generated_states,
                        contradiction_count=contradiction_count,
                        max_frontier_size=max_frontier_size,
                        best_assigned_count=next_state.assigned_count,
                    )
                    return next_state, next_history, engine.cache_info(), diagnostics

                evidence_key = tuple(sorted(next_state.evidence.items()))
                if evidence_key in visited_evidence:
                    continue
                visited_evidence.add(evidence_key)

                heapq.heappush(
                    frontier,
                    FrontierItem(
                        priority=-_frontier_score(next_state, len(next_history)),
                        order=next(queue_counter),
                        state=next_state,
                        history=next_history,
                    ),
                )
                if len(frontier) > frontier_size:
                    heapq.heappop(frontier)
                max_frontier_size = max(max_frontier_size, len(frontier))

    stop_reason = "max_expansions" if expansions >= max_expansions else "frontier_empty"
    diagnostics = FrontierDiagnostics(
        stop_reason=stop_reason,
        expansions=expansions,
        generated_states=generated_states,
        contradiction_count=contradiction_count,
        max_frontier_size=max_frontier_size,
        best_assigned_count=best_state.assigned_count,
    )
    return best_state, best_history, engine.cache_info(), diagnostics


if __name__ == "__main__":
    task = build_sakana_fish_constraint_sudoku_task()

    start = time()
    final_state, history, cache_info, diagnostics = solve_task_with_frontier_search(
        task,
        simulations=100,
        max_expansions=1000,
        rollout_depth=5,
        expand_action_count=5,
        frontier_size=5000,
    )
    print(f"Solved in {time() - start:.2f} seconds")
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
    print("Expansions:", diagnostics.expansions)
    print("Generated states:", diagnostics.generated_states)
    print("Contradictions:", diagnostics.contradiction_count)
    print("Max frontier size:", diagnostics.max_frontier_size)
    print("Frontier:", diagnostics)
