from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .actions import AddClusterAction, AlphaReasonAction, AssignValueAction
from .closure_engine import AlphaReasonClosureEngine
from .state import AlphaReasonState
from .task import AlphaReasonTask


@dataclass
class DFSResult:
    state: AlphaReasonState
    history: List[AlphaReasonAction]
    nodes: int
    solved: bool


def _branch_features(
    task: AlphaReasonTask,
    state: AlphaReasonState,
    branch_feature_count: int,
) -> List[Tuple[str, Tuple[int, ...]]]:
    direct_neighbor_counts = task.metadata.get("direct_neighbor_counts", {})
    return sorted(
        (
            (feature_key, state.feature_supports[feature_key])
            for feature_key in task.target_feature_keys
            if len(state.feature_supports.get(feature_key, ())) > 1
        ),
        key=lambda item: (len(item[1]), -direct_neighbor_counts.get(item[0], 0), item[0]),
    )[:branch_feature_count]


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
        history: List[AlphaReasonAction],
    ) -> Tuple[Optional[AlphaReasonState], List[AlphaReasonAction]]:
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

        for candidate in state.cluster_candidates:
            action = AddClusterAction(candidate.key)
            if self.trace:
                print(
                    f"{len(history) + 1}. Activate {candidate.tier} cluster {candidate.key} "
                    f"(gain={candidate.estimated_gain:.1f}, colors={candidate.colors}, nodes={self.nodes})",
                    flush=True,
                )
            active_clusters = dict(state.active_inference_clusters)
            active_clusters[candidate.key] = candidate.colors
            next_state = self.closure_engine.close(task, state.evidence, active_clusters)
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
                print(f"   Backtrack cluster {candidate.key}", flush=True)

        branch_features = _branch_features(task, state, self.branch_feature_count)

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


class DelayedParallelDFSSolver:
    """
    Tries the best root assignment branch serially first. Only if that branch
    fails are the remaining root sibling branches searched in parallel.
    """

    def __init__(
        self,
        closure_engine_factory: Callable[[], AlphaReasonClosureEngine],
        max_nodes: int = 10000,
        branch_feature_count: int = 1,
        max_workers: int = 2,
        trace: bool = False,
    ):
        self.closure_engine_factory = closure_engine_factory
        self.max_nodes = max_nodes
        self.branch_feature_count = branch_feature_count
        self.max_workers = max_workers
        self.trace = trace
        self.nodes = 0

    def solve(self, task: AlphaReasonTask) -> DFSResult:
        root_solver = AlphaReasonDFSSolver(
            closure_engine=self.closure_engine_factory(),
            max_nodes=self.max_nodes,
            branch_feature_count=self.branch_feature_count,
            trace=self.trace,
        )
        root_state = root_solver.closure_engine.close(
            task,
            task.initial_evidence,
            active_inference_clusters={},
        )
        if self.trace:
            print(
                "0. Forward chaining closure -> "
                f"assigned={root_state.assigned_count}/{len(task.target_feature_keys)}, "
                f"contradiction={root_state.contradiction}, solved={root_state.solved}, "
                f"active_clusters={len(root_state.active_inference_clusters)}",
                flush=True,
            )

        if root_state.contradiction or root_state.solved or root_state.cluster_candidates:
            root_solver.best_state = root_state
            solved_state, history = root_solver._search(task, root_state, [])
            return DFSResult(
                state=solved_state or root_solver.best_state,
                history=history,
                nodes=root_solver.nodes,
                solved=bool(solved_state and solved_state.solved),
            )

        branch_features = _branch_features(task, root_state, self.branch_feature_count)
        if not branch_features:
            root_solver.best_state = root_state
            solved_state, history = root_solver._search(task, root_state, [])
            return DFSResult(
                state=solved_state or root_solver.best_state,
                history=history,
                nodes=root_solver.nodes,
                solved=bool(solved_state and solved_state.solved),
            )

        feature_key, support = branch_features[0]
        ordered_values = sorted(
            support,
            key=lambda value_index: (-root_state.feature_priors[feature_key][value_index], value_index),
        )
        if len(ordered_values) <= 1:
            root_solver.best_state = root_state
            solved_state, history = root_solver._search(task, root_state, [])
            return DFSResult(
                state=solved_state or root_solver.best_state,
                history=history,
                nodes=root_solver.nodes,
                solved=bool(solved_state and solved_state.solved),
            )

        branch_args = [
            (task, root_state, feature_key, support, value_index)
            for value_index in ordered_values
        ]
        first_result = self._solve_branch(branch_args[0])
        if first_result.solved:
            self.nodes = 1 + first_result.nodes
            return DFSResult(
                state=first_result.state,
                history=first_result.history,
                nodes=self.nodes,
                solved=True,
            )

        remaining_branch_args = branch_args[1:]
        if not remaining_branch_args:
            self.nodes = 1 + first_result.nodes
            if root_state.assigned_count >= first_result.state.assigned_count:
                return DFSResult(root_state, [], self.nodes, False)
            return DFSResult(first_result.state, first_result.history, self.nodes, False)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            branch_results = list(executor.map(self._solve_branch, remaining_branch_args))

        branch_results = [first_result, *branch_results]
        self.nodes = 1 + sum(result.nodes for result in branch_results)
        for result in branch_results:
            if result.solved:
                return DFSResult(
                    state=result.state,
                    history=result.history,
                    nodes=self.nodes,
                    solved=True,
                )

        best_result = max(
            branch_results,
            key=lambda result: result.state.assigned_count,
            default=DFSResult(root_state, [], 1, False),
        )
        if root_state.assigned_count >= best_result.state.assigned_count:
            return DFSResult(root_state, [], self.nodes, False)
        return DFSResult(best_result.state, best_result.history, self.nodes, False)

    def _solve_branch(self, branch_arg) -> DFSResult:
        task, root_state, feature_key, support, value_index = branch_arg
        action = AssignValueAction(feature_key, value_index)
        if self.trace:
            print(
                f"1. Try {feature_key}={value_index + 1} "
                f"(support={tuple(value + 1 for value in support)})",
                flush=True,
            )
        solver = AlphaReasonDFSSolver(
            closure_engine=self.closure_engine_factory(),
            max_nodes=self.max_nodes,
            branch_feature_count=self.branch_feature_count,
            trace=self.trace,
        )
        evidence = dict(root_state.evidence)
        evidence.update(task.assignment_to_evidence(feature_key, value_index))
        next_state = solver.closure_engine.close(
            task,
            evidence,
            dict(root_state.active_inference_clusters),
        )
        solver.best_state = next_state
        solved_state, history = solver._search(task, next_state, [action])
        return DFSResult(
            state=solved_state or solver.best_state,
            history=history,
            nodes=solver.nodes,
            solved=bool(solved_state and solved_state.solved),
        )
