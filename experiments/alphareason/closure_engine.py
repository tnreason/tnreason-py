from collections import OrderedDict
from copy import deepcopy
import math
from typing import Dict, Iterable, List, Optional, Tuple

from experiments.constraint_networks.forward_chaining import GenericForwardChaining, check_local_satisfiability
from tnreason import engine

from .actions import AddClusterAction, AssignValueAction, cluster_key_from_colors
from .constraint_propagation import (
    add_cluster_summary_core,
    add_singleton_domain_cores,
    enforce_generalized_arc_consistency,
)
from .cluster_proposals import propose_direct_micro_clusters
from .state import AlphaReasonState, ClusterCandidate
from .task import AlphaReasonTask
from .validation import check_assignment_consistency


def _support_indices(core, feature_key: str, domain_size: int) -> List[int]:
    supported = []
    for idx in range(domain_size):
        try:
            if core[{feature_key: idx}] > 0:
                supported.append(idx)
        except Exception:
            continue
    return supported


def _distribution_from_meanparams(mean_params: Dict[str, object], feature_key: str, domain_size: int) -> Tuple[float, ...]:
    if feature_key not in mean_params:
        return tuple(1.0 / domain_size for _ in range(domain_size))

    values = []
    for idx in range(domain_size):
        try:
            values.append(max(0.0, float(mean_params[feature_key][{feature_key: idx}])))
        except Exception:
            values.append(0.0)

    total = sum(values)
    if total <= 0.0:
        return tuple(1.0 / domain_size for _ in range(domain_size))
    return tuple(value / total for value in values)


def _distribution_from_core(core, feature_key: str, domain_size: int) -> Tuple[float, ...]:
    values = []
    for idx in range(domain_size):
        try:
            values.append(max(0.0, float(core[{feature_key: idx}])))
        except Exception:
            values.append(0.0)

    total = sum(values)
    if total <= 0.0:
        return tuple(1.0 / domain_size for _ in range(domain_size))
    return tuple(value / total for value in values)


class AlphaReasonClosureEngine:
    def __init__(
        self,
        max_message_count: int = 30000,
        cache_size: int = 512,
        allow_cleaning: bool = False,
        enable_micro_clusters: bool = False,
        forward_chaining_parallel: bool = True,
        forward_chaining_max_workers: int = 3,
        forward_chaining_min_parallel_edges: int = 2,
    ):
        self.max_message_count = max_message_count
        self.cache_size = cache_size
        self.allow_cleaning = allow_cleaning
        self.enable_micro_clusters = enable_micro_clusters
        self.forward_chaining_parallel = forward_chaining_parallel
        self.forward_chaining_max_workers = forward_chaining_max_workers
        self.forward_chaining_min_parallel_edges = forward_chaining_min_parallel_edges
        self._closure_cache: "OrderedDict[Tuple[str, Tuple[Tuple[str, int], ...], Tuple[Tuple[str, Tuple[str, ...]], ...]], AlphaReasonState]" = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def close(
        self,
        task: AlphaReasonTask,
        evidence: Dict[str, int],
        active_inference_clusters: Optional[Dict[str, Tuple[str, ...]]] = None,
    ) -> AlphaReasonState:
        active_inference_clusters = dict(active_inference_clusters or {})
        cache_key = self._cache_key(task, evidence, active_inference_clusters)
        if cache_key in self._closure_cache:
            self.cache_hits += 1
            cached_state = self._closure_cache.pop(cache_key)
            self._closure_cache[cache_key] = cached_state
            return deepcopy(cached_state)

        self.cache_misses += 1
        state = task.closure_function(self, task, evidence, active_inference_clusters)
        self._store_cache(cache_key, state)
        return deepcopy(state)

    def cache_info(self) -> Dict[str, int]:
        return {
            "size": len(self._closure_cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "max_size": self.cache_size,
        }

    def _assignment_actions(
        self,
        task: AlphaReasonTask,
        unresolved_features: Iterable[Tuple[str, List[int]]],
        feature_priors: Dict[str, Tuple[float, ...]],
    ) -> Tuple[Tuple[AssignValueAction, ...], Dict[AssignValueAction, float]]:
        candidates = [
            (feature_key, supported)
            for feature_key, supported in unresolved_features
            if len(supported) > 1
        ]
        if not candidates:
            return tuple(), {}

        selected = sorted(
            candidates,
            key=lambda item: (
                len(item[1]),
                -max(feature_priors[item[0]][value_index] for value_index in item[1]),
                item[0],
            ),
        )

        actions = tuple(
            AssignValueAction(feature_key, value_index)
            for feature_key, supported in selected
            for value_index in supported
        )
        priors = {
            action: feature_priors[action.feature_key][action.value_index]
            for action in actions
        }
        return actions, priors

    def close_canetwork(
        self,
        task: AlphaReasonTask,
        evidence: Dict[str, int],
        active_inference_clusters: Dict[str, Tuple[str, ...]],
    ) -> AlphaReasonState:
        from .inference import get_forward_mp_inferer_from_canetwork

        ca_network = task.network_factory(evidence)
        propagator = get_forward_mp_inferer_from_canetwork(
            ca_network,
            allow_cleaning=self.allow_cleaning,
        )

        if active_inference_clusters:
            merged = {
                **propagator.inferenceClusters,
                **{key: list(colors) for key, colors in active_inference_clusters.items()},
            }
            propagator.update_inferenceClusters(merged)

        contradiction = False
        try:
            propagator.propagate_until_convergence(
                nonTrivialFeatureKeys=list(evidence.keys()),
                maxMessageCount=self.max_message_count,
                verbose=False,
            )
        except Exception:
            contradiction = True

        closed_evidence = dict(evidence)
        target_assignments: Dict[str, int] = {}
        feature_supports: Dict[str, Tuple[int, ...]] = {}
        feature_priors: Dict[str, Tuple[float, ...]] = {}
        unresolved_features: List[Tuple[str, List[int]]] = []

        for feature_key in task.target_feature_keys:
            domain_size = task.feature_domain_sizes[feature_key]
            feature_priors[feature_key] = _distribution_from_meanparams(
                getattr(propagator, "meanParamDict", {}),
                feature_key,
                domain_size,
            )
            supported = _support_indices(propagator.caNetwork.canParamDict[feature_key], feature_key, domain_size)
            feature_supports[feature_key] = tuple(supported)
            if len(supported) == 1:
                value_index = supported[0]
                target_assignments[feature_key] = value_index
                closed_evidence.update(task.assignment_to_evidence(feature_key, value_index))
            elif len(supported) == 0:
                contradiction = True
            else:
                unresolved_features.append((feature_key, supported))

        consistency_checker = task.consistency_checker or check_assignment_consistency
        consistency_ok = False
        if not contradiction:
            try:
                consistency_ok = consistency_checker(propagator.caNetwork, closed_evidence)
            except Exception:
                contradiction = True
                consistency_ok = False

        return self._build_state(
            task=task,
            evidence=evidence,
            active_inference_clusters=active_inference_clusters,
            target_assignments=target_assignments,
            feature_supports=feature_supports,
            feature_priors=feature_priors,
            unresolved_features=unresolved_features,
            contradiction=contradiction,
            consistency_ok=consistency_ok,
            propagator=propagator,
            message_count=getattr(propagator, "messageCount", 0),
        )

    def close_constraint_network(
        self,
        task: AlphaReasonTask,
        evidence: Dict[str, int],
        active_inference_clusters: Dict[str, Tuple[str, ...]],
    ) -> AlphaReasonState:
        contradiction = False
        core_dict = task.network_factory(evidence)
        for cluster_key, open_colors in active_inference_clusters.items():
            core_dict = add_cluster_summary_core(
                core_dict,
                open_colors=open_colors,
                name=f"summary_{cluster_key}",
            )
        domains = {}
        try:
            domains, contradiction = enforce_generalized_arc_consistency(core_dict)
            if not contradiction:
                core_dict = add_singleton_domain_cores(core_dict, domains)
        except Exception:
            contradiction = True

        chainer = GenericForwardChaining(
            core_dict,
            parallel=self.forward_chaining_parallel,
            max_workers=self.forward_chaining_max_workers,
            min_parallel_edges=self.forward_chaining_min_parallel_edges,
        )
        try:
            if not contradiction:
                chainer.propagate_all_singleNodeEdges()
        except Exception:
            contradiction = True

        closed_evidence = dict(evidence)
        target_assignments: Dict[str, int] = {}
        feature_supports: Dict[str, Tuple[int, ...]] = {}
        feature_priors: Dict[str, Tuple[float, ...]] = {}
        unresolved_features: List[Tuple[str, List[int]]] = []

        if not contradiction and not chainer.consistent:
            contradiction = True

        for feature_key in task.available_action_feature_keys():
            domain_size = task.feature_domain_sizes[feature_key]
            summary_core = self._summarize_constraint_node(chainer, feature_key)
            if summary_core is None:
                supported = list(domains.get(feature_key, ()))
                if not supported:
                    supported = self._support_from_evidence(task, closed_evidence, feature_key, domain_size)
                if supported:
                    feature_priors[feature_key] = self._distribution_from_supported(domain_size, supported)
                else:
                    feature_priors[feature_key] = tuple(1.0 / domain_size for _ in range(domain_size))
                    supported = list(range(domain_size))
            else:
                feature_priors[feature_key] = _distribution_from_core(summary_core, feature_key, domain_size)
                supported = _support_indices(summary_core, feature_key, domain_size)
                if feature_key in domains:
                    supported = [value_index for value_index in supported if value_index in domains[feature_key]]
                    feature_priors[feature_key] = self._distribution_from_supported(domain_size, supported)
            feature_supports[feature_key] = tuple(supported)

            if len(supported) == 1:
                value_index = supported[0]
                if feature_key in task.target_feature_keys:
                    target_assignments[feature_key] = value_index
                closed_evidence.update(task.assignment_to_evidence(feature_key, value_index))
            elif len(supported) == 0:
                contradiction = True
            else:
                unresolved_features.append((feature_key, supported))

        consistency_checker = task.consistency_checker or (
            lambda constraint_network, assignment: check_local_satisfiability(constraint_network.coresDict)
        )
        consistency_ok = False
        if not contradiction:
            try:
                consistency_ok = consistency_checker(chainer.cn, closed_evidence)
            except Exception:
                contradiction = True
                consistency_ok = False

        cluster_candidates = tuple()
        if self.enable_micro_clusters and not contradiction:
            cluster_candidates = propose_direct_micro_clusters(
                core_dict,
                self._state_preview(
                    task,
                    closed_evidence,
                    target_assignments,
                    feature_supports,
                    feature_priors,
                    active_inference_clusters,
                ),
            )

        return self._build_state(
            task=task,
            evidence=closed_evidence,
            active_inference_clusters=active_inference_clusters,
            target_assignments=target_assignments,
            feature_supports=feature_supports,
            feature_priors=feature_priors,
            unresolved_features=unresolved_features,
            contradiction=contradiction,
            consistency_ok=consistency_ok,
            propagator=None,
            message_count=0,
            cluster_candidates=cluster_candidates,
        )

    def _build_state(
        self,
        task: AlphaReasonTask,
        evidence: Dict[str, int],
        active_inference_clusters: Dict[str, Tuple[str, ...]],
        target_assignments: Dict[str, int],
        feature_supports: Dict[str, Tuple[int, ...]],
        feature_priors: Dict[str, Tuple[float, ...]],
        unresolved_features: List[Tuple[str, List[int]]],
        contradiction: bool,
        consistency_ok: bool,
        propagator,
        message_count: int,
        cluster_candidates: Tuple[ClusterCandidate, ...] = tuple(),
    ) -> AlphaReasonState:
        solved = consistency_ok and len(target_assignments) == len(task.target_feature_keys)
        legal_actions: Tuple[AssignValueAction | AddClusterAction, ...] = tuple()
        action_priors = {}
        if not contradiction and not solved:
            legal_actions, action_priors = self._assignment_actions(task, unresolved_features, feature_priors)
            if propagator is not None:
                cluster_candidates = self._cluster_candidates(task, propagator, active_inference_clusters)
            if not legal_actions and not cluster_candidates:
                contradiction = True

        all_actions = list(legal_actions)
        all_actions.extend(AddClusterAction(candidate.key) for candidate in cluster_candidates)
        for candidate in cluster_candidates:
            action = AddClusterAction(candidate.key)
            action_priors[action] = action_priors.get(action, 0.0)

        closed_evidence = dict(evidence)
        for feature_key, value_index in target_assignments.items():
            closed_evidence.update(task.assignment_to_evidence(feature_key, value_index))

        return AlphaReasonState(
            task_name=task.name,
            evidence=closed_evidence,
            target_feature_keys=task.target_feature_keys,
            feature_domain_sizes=task.feature_domain_sizes,
            target_assignments=target_assignments,
            feature_supports=feature_supports,
            feature_priors=feature_priors,
            legal_actions=tuple(all_actions),
            action_priors=self._normalize_action_priors(action_priors, tuple(all_actions)),
            active_inference_clusters=active_inference_clusters,
            cluster_candidates=cluster_candidates,
            contradiction=contradiction,
            consistency_ok=consistency_ok,
            solved=solved,
            value_estimate=self._value_estimate(task, feature_priors, target_assignments, contradiction, solved),
            assigned_count=len(target_assignments),
            message_count=message_count,
        )

    def _cluster_candidates(
        self,
        task: AlphaReasonTask,
        propagator,
        active_inference_clusters: Dict[str, Tuple[str, ...]],
    ) -> Tuple[ClusterCandidate, ...]:
        candidates = []
        seen = set(active_inference_clusters.keys())
        for detector in task.rule_detectors:
            try:
                new_clusters = detector(propagator, **task.rule_detector_kwargs)
            except Exception:
                continue
            for colors in new_clusters:
                colors_tuple = tuple(colors)
                key = cluster_key_from_colors(colors_tuple)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(ClusterCandidate(key=key, colors=colors_tuple))
        return tuple(sorted(candidates, key=lambda candidate: candidate.key))

    def _state_preview(
        self,
        task: AlphaReasonTask,
        evidence: Dict[str, int],
        target_assignments: Dict[str, int],
        feature_supports: Dict[str, Tuple[int, ...]],
        feature_priors: Dict[str, Tuple[float, ...]],
        active_inference_clusters: Dict[str, Tuple[str, ...]],
    ) -> AlphaReasonState:
        return AlphaReasonState(
            task_name=task.name,
            evidence=evidence,
            target_feature_keys=task.target_feature_keys,
            feature_domain_sizes=task.feature_domain_sizes,
            target_assignments=target_assignments,
            feature_supports=feature_supports,
            feature_priors=feature_priors,
            active_inference_clusters=active_inference_clusters,
        )

    def _summarize_constraint_node(self, chainer: GenericForwardChaining, feature_key: str):
        edge_keys = [
            edge_key
            for edge_key, core in chainer.cn.coresDict.items()
            if feature_key in core.colors
        ]
        if not edge_keys:
            return None
        return engine.contract(
            {edge_key: chainer.cn.coresDict[edge_key] for edge_key in edge_keys},
            openColors=[feature_key],
        )

    def _support_from_evidence(
        self,
        task: AlphaReasonTask,
        evidence: Dict[str, int],
        feature_key: str,
        domain_size: int,
    ) -> List[int]:
        supported = []
        for value_index in range(domain_size):
            if evidence.get(feature_key) == value_index:
                supported.append(value_index)
                continue
            assignment_evidence = task.assignment_to_evidence(feature_key, value_index)
            if assignment_evidence and any(evidence.get(key) == value for key, value in assignment_evidence.items()):
                supported.append(value_index)
        return supported

    def _distribution_from_supported(self, domain_size: int, supported: List[int]) -> Tuple[float, ...]:
        if not supported:
            return tuple(1.0 / domain_size for _ in range(domain_size))
        probability = 1.0 / len(supported)
        return tuple(probability if idx in supported else 0.0 for idx in range(domain_size))

    def _value_estimate(
        self,
        task: AlphaReasonTask,
        feature_priors: Dict[str, Tuple[float, ...]],
        target_assignments: Dict[str, int],
        contradiction: bool,
        solved: bool,
    ) -> float:
        if contradiction:
            return -1.0
        if solved:
            return 1.0

        entropies = []
        for feature_key in task.target_feature_keys:
            if feature_key in target_assignments:
                continue
            probs = feature_priors[feature_key]
            entropy = -sum(prob * math.log(prob + 1e-12) for prob in probs)
            entropies.append(entropy)

        if not entropies:
            return 1.0

        max_entropy = max(math.log(task.feature_domain_sizes[feature_key]) for feature_key in task.target_feature_keys)
        avg_entropy = sum(entropies) / len(entropies)
        entropy_score = 1.0 - (avg_entropy / max_entropy if max_entropy > 0 else 0.0)
        coverage_score = len(target_assignments) / max(1, len(task.target_feature_keys))
        value = 0.5 * (2.0 * entropy_score - 1.0) + 0.5 * (2.0 * coverage_score - 1.0)
        return max(-1.0, min(1.0, value))

    def _normalize_action_priors(self, priors: Dict[object, float], actions: Tuple[object, ...]) -> Dict[object, float]:
        if not actions:
            return {}
        filtered = {action: max(0.0, float(priors.get(action, 0.0))) for action in actions}
        total = sum(filtered.values())
        if total <= 0:
            uniform = 1.0 / len(actions)
            return {action: uniform for action in actions}
        return {action: value / total for action, value in filtered.items()}

    def _cache_key(
        self,
        task: AlphaReasonTask,
        evidence: Dict[str, int],
        active_inference_clusters: Dict[str, Tuple[str, ...]],
    ) -> Tuple[str, Tuple[Tuple[str, int], ...], Tuple[Tuple[str, Tuple[str, ...]], ...]]:
        return (
            task.name,
            self.enable_micro_clusters,
            tuple(task.available_action_feature_keys()),
            tuple(sorted(evidence.items())),
            tuple(sorted((key, tuple(colors)) for key, colors in active_inference_clusters.items())),
        )

    def _store_cache(self, cache_key, state: AlphaReasonState) -> None:
        self._closure_cache[cache_key] = deepcopy(state)
        self._closure_cache.move_to_end(cache_key)
        while len(self._closure_cache) > self.cache_size:
            self._closure_cache.popitem(last=False)
