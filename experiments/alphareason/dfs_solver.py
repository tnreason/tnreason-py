from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .actions import AssignValueAction
from .closure_engine import AlphaReasonClosureEngine
from .state import AlphaReasonState
from .task import AlphaReasonTask


@dataclass
class DFSResult:
    state: AlphaReasonState
    history: List[AssignValueAction]
    nodes: int
    solved: bool


class AlphaReasonDFSSolver:
    def __init__(
        self,
        closure_engine: AlphaReasonClosureEngine,
        max_nodes: int = 10000,
        branch_feature_count: int = 1,
        trace: bool = False,
    ):
        self.closure_engine = closure_engine
        self.max_nodes = max_nodes
        self.branch_feature_count = branch_feature_count
        self.trace = trace
        self.nodes = 0
        self.failed: set[Tuple[Tuple[Tuple[str, int], ...], Tuple[Tuple[str, Tuple[str, ...]], ...]]] = set()

    def solve(self, task: AlphaReasonTask) -> DFSResult:
        self.nodes = 0
        self.failed.clear()
        state = self.closure_engine.close(task, task.initial_evidence, active_inference_clusters={})
        self.best_state = state
        if self.trace:
            print(
                "0. Forward chaining closure -> "
                f"assigned={state.assigned_count}/{len(task.target_feature_keys)}, "
                f"contradiction={state.contradiction}, solved={state.solved}, "
                f"active_clusters={len(state.active_inference_clusters)}",
                flush=True,
            )
        solved_state, history = self._search(task, state, [])
        return DFSResult(
            state=solved_state or self.best_state,
            history=history,
            nodes=self.nodes,
            solved=bool(solved_state and solved_state.solved),
        )

    def _search(
        self,
        task: AlphaReasonTask,
        state: AlphaReasonState,
        history: List[AssignValueAction],
    ) -> Tuple[Optional[AlphaReasonState], List[AssignValueAction]]:
        self.nodes += 1
        if self.nodes > self.max_nodes or state.contradiction:
            return None, history
        if state.solved:
            return state, history
        if state.assigned_count > self.best_state.assigned_count:
            self.best_state = state

        state_key = state.state_key()
        if state_key in self.failed:
            return None, history

        direct_neighbor_counts = task.metadata.get("direct_neighbor_counts", {})
        branch_features = sorted(
            (
                (feature_key, state.feature_supports[feature_key])
                for feature_key in task.target_feature_keys
                if len(state.feature_supports.get(feature_key, ())) > 1
            ),
            key=lambda item: (len(item[1]), -direct_neighbor_counts.get(item[0], 0), item[0]),
        )[: self.branch_feature_count]

        for feature_key, support in branch_features:
            ordered_values = sorted(
                support,
                key=lambda value_index: (-state.feature_priors[feature_key][value_index], value_index),
            )
            for value_index in ordered_values:
                action = AssignValueAction(feature_key, value_index)
                if self.trace:
                    print(
                        f"{len(history) + 1}. Try {feature_key}={value_index + 1} "
                        f"(support={tuple(value + 1 for value in support)}, nodes={self.nodes})",
                        flush=True,
                    )
                evidence = dict(state.evidence)
                evidence.update(task.assignment_to_evidence(feature_key, value_index))
                next_state = self.closure_engine.close(task, evidence, state.active_inference_clusters)
                if self.trace:
                    print(
                        "   Forward chaining closure -> "
                        f"assigned={next_state.assigned_count}/{len(task.target_feature_keys)}, "
                        f"contradiction={next_state.contradiction}, solved={next_state.solved}, "
                        f"active_clusters={len(next_state.active_inference_clusters)}",
                        flush=True,
                    )
                solved_state, solved_history = self._search(task, next_state, [*history, action])
                if solved_state is not None:
                    return solved_state, solved_history
                if self.trace:
                    print(f"   Backtrack {feature_key}={value_index + 1}", flush=True)

        self.failed.add(state_key)
        return None, history
