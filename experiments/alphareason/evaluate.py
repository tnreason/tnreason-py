from dataclasses import asdict, dataclass
from typing import Dict, Optional

from .policies import HeuristicPolicyValue, TrainablePolicyValue
from .run_experiment import solve_task
from .state import AlphaReasonState
from .task import AlphaReasonTask
from .train import (
    TrainingConfig,
    load_hf_sudoku_dataset_training_tasks,
    train_supervised,
)


@dataclass
class EvaluationResult:
    label: str
    solved: bool
    contradiction: bool
    assigned_count: int
    target_count: int
    correct_assigned_count: int
    exact_match: bool
    history_length: int
    cache_hits: int
    cache_misses: int


def summarize_state(
    label: str,
    task: AlphaReasonTask,
    state: AlphaReasonState,
    history,
    cache_info: Dict[str, int],
) -> EvaluationResult:
    correct_assigned_count = sum(
        1
        for feature_key, value_index in state.target_assignments.items()
        if task.solution_assignments.get(feature_key) == value_index
    )
    exact_match = (
        len(state.target_assignments) == len(task.solution_assignments)
        and all(task.solution_assignments.get(feature_key) == value_index for feature_key, value_index in state.target_assignments.items())
    )
    return EvaluationResult(
        label=label,
        solved=state.solved,
        contradiction=state.contradiction,
        assigned_count=state.assigned_count,
        target_count=len(task.target_feature_keys),
        correct_assigned_count=correct_assigned_count,
        exact_match=exact_match,
        history_length=len(history),
        cache_hits=cache_info.get("hits", 0),
        cache_misses=cache_info.get("misses", 0),
    )


def compare_heuristic_and_trained_on_validation(
    train_limit: int = 8,
    valid_index: int = 0,
    training_config: Optional[TrainingConfig] = None,
    simulations: int = 20,
    max_steps: int = 30,
):
    train_tasks = load_hf_sudoku_dataset_training_tasks(limit=train_limit, offset=0, split="train")
    valid_tasks = load_hf_sudoku_dataset_training_tasks(limit=1, offset=valid_index, split="valid")
    if not valid_tasks:
        raise ValueError("No validation task could be loaded.")
    valid_task = valid_tasks[0]

    training_config = training_config or TrainingConfig(
        epochs=2,
        learning_rate=1e-3,
        max_samples_per_task=4,
        seed=0,
    )
    model, history = train_supervised(train_tasks, config=training_config)

    heuristic_state, heuristic_history, heuristic_cache = solve_task(
        valid_task,
        policy_value=HeuristicPolicyValue(),
        simulations=simulations,
        max_steps=max_steps,
    )
    trained_policy = TrainablePolicyValue(model=model, policy_mix=1.0, value_mix=1.0)
    trained_state, trained_history, trained_cache = solve_task(
        valid_task,
        policy_value=trained_policy,
        simulations=simulations,
        max_steps=max_steps,
    )

    return {
        "training_history": history,
        "validation_task": {
            "name": valid_task.name,
            "metadata": dict(valid_task.metadata),
        },
        "heuristic": asdict(summarize_state("heuristic", valid_task, heuristic_state, heuristic_history, heuristic_cache)),
        "trained": asdict(summarize_state("trained", valid_task, trained_state, trained_history, trained_cache)),
    }


if __name__ == "__main__":
    result = compare_heuristic_and_trained_on_validation()
    print("Validation task:", result["validation_task"])
    print("Final training epoch:", result["training_history"][-1])
    print("Heuristic:", result["heuristic"])
    print("Trained:", result["trained"])
