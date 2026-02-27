from collections import OrderedDict
from copy import deepcopy
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import demonstrations.sudoku.inferenceClusters as inf_clusters

from .actions import ClusterCandidate, cluster_key_from_colors
from .inference import get_forward_mp_inferer_from_canetwork
from .state import AlphaSudokuState
from .validation import check_assignment_consistency


def _all_cell_ids(num: int) -> List[Tuple[int, int, int, int]]:
    return [
        (r1, r2, c1, c2)
        for r1 in range(num)
        for r2 in range(num)
        for c1 in range(num)
        for c2 in range(num)
    ]


def _cell_id_from_atom(atom_key: str) -> Optional[Tuple[int, int, int, int]]:
    if not atom_key.startswith("a_"):
        return None
    parts = atom_key.split("_")
    if len(parts) != 6:
        return None
    return tuple(int(x) for x in parts[1:5])


def mean_params_to_evidence(mean_params: Dict[str, object]) -> Dict[str, int]:
    return {
        feature_key: 1
        for feature_key in mean_params
        if feature_key.startswith("a_") and mean_params[feature_key][{feature_key: 0}] == 0
    }


def atom_prior_from_meanparams(mean_params: Dict[str, object], atom_key: str) -> float:
    try:
        core = mean_params[atom_key]
        p1 = float(core[{atom_key: 1}])
        p0 = float(core[{atom_key: 0}])
    except Exception:
        return 0.0
    total = p0 + p1
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, p1 / total))


class AlphaSudokuClosureEngine:
    def __init__(
        self,
        num: int = 3,
        canetwork_factory: Optional[Callable[[Dict[str, int]], object]] = None,
        rule_detectors: Optional[Iterable[Callable[[object, int], List[List[str]]]]] = None,
        max_message_count: int = 30000,
        cache_size: int = 512,
        allow_cleaning: bool = False,
    ):
        self.num = num
        self.canetwork_factory = canetwork_factory
        self.max_message_count = max_message_count
        self.cache_size = cache_size
        self.allow_cleaning = allow_cleaning
        self.rule_detectors = list(rule_detectors or [
            inf_clusters.detect_locked_candidates_square,
            inf_clusters.detect_locked_candidates_row,
            inf_clusters.detect_locked_candidates_col,
            inf_clusters.detect_hidden_pairs,
            inf_clusters.detect_naked_pairs,
        ])
        self._closure_cache: "OrderedDict[Tuple[Tuple[Tuple[str, int], ...], Tuple[Tuple[str, Tuple[str, ...]], ...]], AlphaSudokuState]" = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def close(
        self,
        evidence: Dict[str, int],
        active_inference_clusters: Optional[Dict[str, Tuple[str, ...]]] = None,
    ) -> AlphaSudokuState:
        active_inference_clusters = dict(active_inference_clusters or {})
        cache_key = self._cache_key(evidence, active_inference_clusters)
        if cache_key in self._closure_cache:
            self.cache_hits += 1
            cached_state = self._closure_cache.pop(cache_key)
            self._closure_cache[cache_key] = cached_state
            return deepcopy(cached_state)

        self.cache_misses += 1
        if self.canetwork_factory is None:
            raise ValueError("AlphaSudokuClosureEngine requires a canetwork_factory.")
        ca_network = self.canetwork_factory(evidence)
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
        if hasattr(propagator, "meanParamDict"):
            try:
                closed_evidence.update(mean_params_to_evidence(propagator.meanParamDict))
            except Exception:
                pass

        assigned_count = len([key for key, value in closed_evidence.items() if key.startswith("a_") and value == 1])

        consistency_ok = False
        if not contradiction:
            try:
                consistency_ok = check_assignment_consistency(propagator.caNetwork, closed_evidence)
            except Exception:
                consistency_ok = False
                contradiction = True

        solved = consistency_ok and assigned_count == (self.num ** 4)

        assignment_actions: Tuple[str, ...] = tuple()
        atom_priors: Dict[str, float] = {}
        cluster_candidates: Tuple[ClusterCandidate, ...] = tuple()
        if not contradiction and not solved:
            assignment_actions = self._assignment_actions(propagator, closed_evidence)
            atom_priors = {
                atom_key: atom_prior_from_meanparams(propagator.meanParamDict, atom_key)
                for atom_key in assignment_actions
            }
            cluster_candidates = self._cluster_candidates(propagator, active_inference_clusters)

        state = AlphaSudokuState(
            evidence=closed_evidence,
            active_inference_clusters=active_inference_clusters,
            assignment_actions=assignment_actions,
            cluster_candidates=cluster_candidates,
            atom_priors=atom_priors,
            contradiction=contradiction,
            consistency_ok=consistency_ok,
            solved=solved,
            value_estimate=self._value_estimate(propagator, closed_evidence, contradiction, solved),
            assigned_count=assigned_count,
            message_count=getattr(propagator, "messageCount", 0),
        )
        self._store_cache(cache_key, state)
        return deepcopy(state)

    def cache_info(self) -> Dict[str, int]:
        return {
            "size": len(self._closure_cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "max_size": self.cache_size,
        }

    def clear_cache(self) -> None:
        self._closure_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def _assignment_actions(self, propagator, evidence: Dict[str, int]) -> Tuple[str, ...]:
        candidates_by_cell = {cid: [] for cid in _all_cell_ids(self.num)}
        decided_cells = {
            _cell_id_from_atom(key)
            for key, value in evidence.items()
            if value == 1 and key.startswith("a_")
        }

        for key, core in propagator.caNetwork.canParamDict.items():
            if not key.startswith("a_") or key in evidence:
                continue
            cell_id = _cell_id_from_atom(key)
            if cell_id is None or cell_id in decided_cells:
                continue
            try:
                if core[{key: 1}] > 0:
                    candidates_by_cell[cell_id].append(key)
            except Exception:
                continue

        best_cell = None
        best_count = None
        for cell_id, cell_candidates in candidates_by_cell.items():
            count = len(cell_candidates)
            if 1 < count and (best_count is None or count < best_count):
                best_cell = cell_id
                best_count = count
        if best_cell is None:
            return tuple()
        return tuple(sorted(candidates_by_cell[best_cell]))

    def _cluster_candidates(
        self,
        propagator,
        active_inference_clusters: Dict[str, Tuple[str, ...]],
    ) -> Tuple[ClusterCandidate, ...]:
        candidates: List[ClusterCandidate] = []
        seen = set(active_inference_clusters.keys())
        for detector in self.rule_detectors:
            try:
                new_clusters = detector(propagator, num=self.num)
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

    def _value_estimate(self, propagator, evidence: Dict[str, int], contradiction: bool, solved: bool) -> float:
        if contradiction:
            return -1.0
        if solved:
            return 1.0

        decided_cells = {
            _cell_id_from_atom(key)
            for key, value in evidence.items()
            if value == 1 and key.startswith("a_")
        }
        entropies = []
        for cell_id in _all_cell_ids(self.num):
            if cell_id in decided_cells:
                continue
            r1, r2, c1, c2 = cell_id
            probs = []
            for digit in range(self.num ** 2):
                atom_key = f"a_{r1}_{r2}_{c1}_{c2}_{digit}"
                probs.append(atom_prior_from_meanparams(propagator.meanParamDict, atom_key))
            total = sum(probs)
            if total <= 1e-12:
                entropies.append(math.log(self.num ** 2))
                continue
            probs = [prob / total for prob in probs]
            entropies.append(-sum(prob * math.log(prob + 1e-12) for prob in probs))

        if not entropies:
            return 1.0

        avg_entropy = sum(entropies) / len(entropies)
        max_entropy = math.log(self.num ** 2)
        entropy_score = 1.0 - (avg_entropy / max_entropy)
        coverage_score = len(decided_cells) / (self.num ** 4)
        value = 0.5 * (2.0 * entropy_score - 1.0) + 0.5 * (2.0 * coverage_score - 1.0)
        return max(-1.0, min(1.0, value))

    def _cache_key(
        self,
        evidence: Dict[str, int],
        active_inference_clusters: Dict[str, Tuple[str, ...]],
    ) -> Tuple[Tuple[Tuple[str, int], ...], Tuple[Tuple[str, Tuple[str, ...]], ...]]:
        return (
            tuple(sorted(evidence.items())),
            tuple(sorted((key, tuple(colors)) for key, colors in active_inference_clusters.items())),
        )

    def _store_cache(
        self,
        cache_key: Tuple[Tuple[Tuple[str, int], ...], Tuple[Tuple[str, Tuple[str, ...]], ...]],
        state: AlphaSudokuState,
    ) -> None:
        self._closure_cache[cache_key] = deepcopy(state)
        self._closure_cache.move_to_end(cache_key)
        while len(self._closure_cache) > self.cache_size:
            self._closure_cache.popitem(last=False)
