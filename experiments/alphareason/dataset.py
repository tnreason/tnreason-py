import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .actions import AssignValueAction
from .closure_engine import AlphaReasonClosureEngine
from .state import AlphaReasonState
from .task import AlphaReasonTask
from .training_targets import policy_target_from_solution, value_target_from_solution


@dataclass
class SupervisedSample:
    task_name: str
    state: AlphaReasonState
    policy_target: Dict[object, float]
    value_target: float


def extract_solution_trajectory_samples(
    task: AlphaReasonTask,
    closure_engine: Optional[AlphaReasonClosureEngine] = None,
    shuffle_good_actions: bool = False,
    rng: Optional[random.Random] = None,
) -> List[SupervisedSample]:
    closure_engine = closure_engine or AlphaReasonClosureEngine()
    rng = rng or random.Random(0)

    evidence = dict(task.initial_evidence)
    state = closure_engine.close(task, evidence, active_inference_clusters={})
    samples: List[SupervisedSample] = []

    while not state.done:
        policy_target = policy_target_from_solution(state, task)
        if policy_target:
            samples.append(
                SupervisedSample(
                    task_name=task.name,
                    state=state,
                    policy_target=policy_target,
                    value_target=value_target_from_solution(state, task),
                )
            )

            good_actions = list(policy_target.keys())
            if shuffle_good_actions:
                rng.shuffle(good_actions)
            chosen_action = good_actions[0]
            if isinstance(chosen_action, AssignValueAction):
                evidence.update(task.assignment_to_evidence(chosen_action.feature_key, chosen_action.value_index))
                state = closure_engine.close(
                    task,
                    evidence,
                    active_inference_clusters=state.active_inference_clusters,
                )
                continue

        break

    return samples


def build_supervised_dataset(
    tasks: Sequence[AlphaReasonTask],
    shuffle_good_actions: bool = False,
    seed: int = 0,
) -> List[SupervisedSample]:
    rng = random.Random(seed)
    closure_engine = AlphaReasonClosureEngine()
    dataset: List[SupervisedSample] = []
    for task in tasks:
        dataset.extend(
            extract_solution_trajectory_samples(
                task,
                closure_engine=closure_engine,
                shuffle_good_actions=shuffle_good_actions,
                rng=rng,
            )
        )
    return dataset
