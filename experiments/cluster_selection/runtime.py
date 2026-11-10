"""Opt-in cluster application at GAC closure, retaining pruning but not summaries."""
from os import PathLike
from time import perf_counter
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from .candidate_generation import ClusterCandidateGenerator
from .candidate_evaluation_mac import prepare_candidate, contract_candidate
from .data import network_data


def load(path):
    """Dispatch old supervised and new construction checkpoints, only when requested."""
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != 1:
        raise ValueError("Expected a version-1 cluster checkpoint")
    config = dict(checkpoint["config"])
    if checkpoint.get("kind") == "construction":
        from .reinforcement.model import Policy as model_type
        config.setdefault("normalize", False)  # Preserve inference for legacy checkpoints.
    elif checkpoint.get("kind") is None:
        from .supervised.model import Predictor as model_type
    else:
        raise ValueError("Unknown cluster checkpoint kind")
    model = model_type(checkpoint["network"], **config)
    model.reward_colors = checkpoint["reward_colors"]
    if checkpoint.get("kind") == "construction":
        model.beta = checkpoint["beta"]
    model.load_state_dict(checkpoint["weights"])
    return model.eval()


def _base_network(state, predictor):
    raw = predictor.network
    keys = [core["key"] for core in raw["cores"]]
    if set(keys) - state._cores.keys():
        raise ValueError("Predictor network cores are missing from the state")
    base = ConstraintNetwork({key: state._cores[key] for key in keys}, maintain_indices=True)
    extras = state._cores.keys() - base.coresDict.keys()
    if network_data(base) != raw or dict(state.dimensions) != base.dimension_dict or any(
            len(state._cores[key].colors) != 1 for key in extras):
        raise ValueError("Predictor schema differs; only additional unary evidence is supported")
    return base


def apply_candidate(state, candidate, prepared=None, **limits):
    """Contract once; retain pruning/trail, but remove the summary even on error."""
    if state.contradiction or state._queue or state._queued:
        raise ValueError("Apply a candidate only after noncontradictory GAC closure")
    key = f"summary_{candidate.key}"
    if key in state._cores:
        raise ValueError("Candidate summary key is already active")
    summary = contract_candidate(candidate, prepare_candidate(state, candidate, **limits) if prepared is None else prepared)
    hook, previously_added = state.on_closure, key in state._added_core_keys
    state.push_level()
    state.on_closure = None
    keep = False
    try:
        state.add_cores({key: summary}, reversible=True)
        result = state.propagate()
        keep = True
        return result
    finally:
        state.pop_level(keep_domains=keep)
        state.on_closure = hook
        if not previously_added:
            state._added_core_keys.discard(key)


class Policy:
    """Bound to one state; cache plans, never domain-dependent results.

    Predictors expose network and select; paths load supervised checkpoints.
    """

    def __init__(self, state, predictor, *, max_candidates=32, max_work=1_000_000, max_size=100_000,
                 cost_weight=1.0, contradiction_reward=1.0):
        if isinstance(predictor, (str, PathLike)):
            from .supervised.model import load
            predictor = load(predictor)
        base = _base_network(state, predictor)
        raw = predictor.network
        keys = [core["key"] for core in raw["cores"]]
        self.state, self.predictor = state, predictor
        self.colors, self.color_idx = raw["colors"], {color: i for i, color in enumerate(raw["colors"])}
        self.core_idx = {key: i for i, key in enumerate(keys)}
        self.generator = ClusterCandidateGenerator(base, max_candidates=max_candidates, max_work=max_work, max_size=max_size)
        self.limits = dict(max_work=max_work, max_size=max_size)
        self.weights = dict(cost_weight=cost_weight, contradiction_reward=contradiction_reward)
        self.plans, self.used, self.last = {}, {}, None
        self.statistics = dict(calls=0, candidates=0, selected=0, removed_values=0, contradictions=0, time_s=0.0)

    def __call__(self, state, changed):
        if state is not self.state:
            raise ValueError("A learned policy is bound to one state")
        if state.contradiction or state.is_solved():
            return
        snapshot = tuple(state.domains[color] for color in self.colors)
        if snapshot == self.last:
            return
        start = perf_counter()
        self.statistics["calls"] += 1
        try:
            candidates, encoded, plans = [], [], []
            for candidate in self.generator.generate(state.domains, changed_colors=changed):
                signature = candidate.core_keys, candidate.open_colors
                colors = sorted(set().union(*(self.generator.scopes[key] for key in candidate.core_keys)))
                if self.used.get(signature) == tuple(state.domains[color] for color in colors):
                    continue
                if signature not in self.plans:
                    if len(self.plans) >= self.generator.max_steps:
                        evicted = next(iter(self.plans))
                        self.plans.pop(evicted)
                        self.used.pop(evicted, None)
                    try:
                        self.plans[signature] = prepare_candidate(state, candidate, **self.limits)
                    except ValueError as error:
                        if str(error) not in ("Candidate exceeds the contraction work or tensor-size budget",
                                              "NumPy contraction supports at most 52 distinct colors"):
                            raise
                        self.plans[signature] = None
                prepared = self.plans[signature]
                if prepared is None:
                    continue
                estimate = prepared[0]
                encoded.append(dict(cores=[self.core_idx[key] for key in candidate.core_keys],
                                    open=[self.color_idx[color] for color in candidate.open_colors],
                                    estimate=[estimate.work, estimate.max_size, estimate.output_size]))
                candidates.append(candidate)
                plans.append(prepared)
            self.statistics["candidates"] += len(candidates)
            idx = self.predictor.select(state.domains, encoded, **self.weights) if candidates else None
            if idx is not None:
                result = apply_candidate(state, candidates[idx], plans[idx])
                self.statistics["selected"] += 1
                self.statistics["removed_values"] += result.removed_values
                self.statistics["contradictions"] += int(result.contradiction)
                if not result.contradiction:
                    chosen = candidates[idx]
                    colors = sorted(set().union(*(self.generator.scopes[key] for key in chosen.core_keys)))
                    self.used[chosen.core_keys, chosen.open_colors] = tuple(state.domains[color] for color in colors)
            self.last = tuple(state.domains[color] for color in self.colors)
        finally:
            self.statistics["time_s"] += perf_counter() - start


def attach(state, predictor, **options):
    """Attach before propagation/search. Existing cluster hooks are not replaced."""
    if state.on_closure is not None:
        raise ValueError("Learned clusters cannot be combined with another closure hook")
    if isinstance(predictor, (str, PathLike)):
        predictor = load(predictor)
    policy = ConstructionPolicy if getattr(predictor, "kind", None) == "construction" else Policy
    state.on_closure = policy(state, predictor, **options)
    return state


class ConstructionPolicy:
    """Apply up to max_candidates affordable constructions at each GAC closure."""

    def __init__(self, state, predictor, *, max_candidates=None, stop_on_noop=False):
        if max_candidates is None:
            max_candidates = predictor.config.get("cluster_attempts", 32)
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates < 1:
            raise ValueError("Construction proposal limit must be a positive integer")
        _base_network(state, predictor)
        self.state, self.predictor, self.colors = state, predictor, predictor.network["colors"]
        self.max_candidates = max_candidates
        self.alternatives = getattr(predictor, "proposals", None) if not stop_on_noop else None
        self.plans, self.used, self.last = {}, {}, None
        self.statistics = dict(calls=0, candidates=0, selected=0, useful=0, rejected=0,
                               skipped=0, removed_values=0, contradictions=0, time_s=0.0)
        self.compact_domains = predictor.config.get("compact_domains", False)
        self.max_total_work = predictor.config.get("max_total_work")
        if self.compact_domains or self.max_total_work is not None:
            self.statistics.update(estimated_work=0, max_tensor_size=0, budget_exhausted=0)

    def __call__(self, state, changed):
        if state is not self.state:
            raise ValueError("A learned policy is bound to one state")
        snapshot = tuple(state.domains[color] for color in self.colors)
        if state.contradiction or state.is_solved() or snapshot == self.last:
            return
        started = perf_counter()
        self.statistics["calls"] += 1
        spent, attempted, proposals = 0, 0, None
        try:
            while not state.contradiction and not state.is_solved():
                if self.alternatives is not None and attempted >= self.max_candidates:
                    break
                if self.max_total_work is not None and spent >= self.max_total_work:
                    self.statistics["budget_exhausted"] += 1
                    break
                if self.alternatives is not None:
                    if proposals is None:
                        # Rejects/cache hits do not consume contraction slots.
                        # The decoder separately bounds construction-state search.
                        proposals = iter(self.alternatives(state.domains,
                            max_candidates=8192))
                    candidate = next(proposals, None)
                else:
                    # Legacy/custom single-proposal adapters keep their stop rule.
                    candidate = self.predictor.propose(state.domains)
                if candidate is None:
                    break
                self.statistics["candidates"] += 1
                signature = candidate.core_keys, candidate.open_colors
                colors = sorted({color for key in candidate.core_keys for color in state._cores[key].colors})
                fingerprint = tuple(state.domains[color] for color in colors)
                if self.used.get(signature) == fingerprint:
                    self.statistics["skipped"] += 1
                    if self.alternatives is not None:
                        continue
                    break
                if self.compact_domains:
                    # Slices and shapes belong to this domain snapshot only.
                    try:
                        limits = {key: self.predictor.config[key] for key in ("max_work", "max_size")}
                        prepared = prepare_candidate(state, candidate, compact_domains=True, **limits)
                    except ValueError as error:
                        if str(error) not in ("Candidate exceeds the contraction work or tensor-size budget",
                                              "NumPy contraction supports at most 52 distinct colors"):
                            raise
                        prepared = None
                elif signature not in self.plans:
                    if len(self.plans) >= 1024:
                        evicted = next(iter(self.plans))
                        self.plans.pop(evicted)
                        self.used.pop(evicted, None)
                    try:
                        limits = {key: self.predictor.config[key] for key in ("max_work", "max_size")}
                        self.plans[signature] = prepare_candidate(state, candidate, **limits)
                    except ValueError as error:
                        if str(error) not in ("Candidate exceeds the contraction work or tensor-size budget",
                                              "NumPy contraction supports at most 52 distinct colors"):
                            raise
                        self.plans[signature] = None
                if not self.compact_domains:
                    prepared = self.plans[signature]
                if prepared is None:
                    self.statistics["rejected"] += 1
                    if self.alternatives is not None:
                        continue
                    break
                estimate = prepared[0]
                if self.max_total_work is not None and spent + estimate.work > self.max_total_work:
                    self.statistics["budget_exhausted"] += 1
                    break
                attempted += 1
                result = apply_candidate(state, candidate, prepared)
                spent += estimate.work
                if "estimated_work" in self.statistics:
                    self.statistics["estimated_work"] += estimate.work
                    self.statistics["max_tensor_size"] = max(self.statistics["max_tensor_size"], estimate.max_size)
                self.statistics["selected"] += 1
                self.statistics["useful"] += int(result.removed_values > 0 or result.contradiction)
                self.statistics["removed_values"] += result.removed_values
                self.statistics["contradictions"] += int(result.contradiction)
                if not result.removed_values:
                    if not result.contradiction:
                        # Cache only no-op inputs; pruning may make the same cluster useful again.
                        if self.compact_domains and len(self.used) >= 1024:
                            self.used.pop(next(iter(self.used)))
                        self.used[signature] = fingerprint
                    if self.alternatives is None:
                        break
                else:
                    # Never reuse an encoding or construction frontier after pruning.
                    proposals = None
            self.last = None if state.contradiction else tuple(state.domains[color] for color in self.colors)
        finally:
            self.statistics["time_s"] += perf_counter() - started


class HeuristicPolicy:
    """Apply a bounded batch sequentially; no predictor or recursive generation."""

    def __init__(self, state, *, max_candidates=32, max_work=1_000_000, max_size=100_000,
                 compact_domains=False, **options):
        self.state, self.colors = state, tuple(state.domains)
        self.compact_domains = compact_domains
        self.generator = ClusterCandidateGenerator(state._cores, max_candidates=max_candidates,
                                                   max_work=max_work, max_size=max_size, **options)
        self.limits = dict(max_work=max_work, max_size=max_size)
        self.plans, self.used, self.last = {}, {}, None
        self.statistics = dict(calls=0, proposed=0, applied=0, useful=0, skipped=0, rejected=0,
                               removed_values=0, contradictions=0, time_s=0.0)

    def __call__(self, state, changed):
        if state is not self.state:
            raise ValueError("A heuristic policy is bound to one state")
        if state.contradiction or state.is_solved():
            return
        snapshot = tuple(state.domains[color] for color in self.colors)
        if snapshot == self.last:
            return
        start = perf_counter()
        self.statistics["calls"] += 1
        try:
            generation = dict(changed_colors=changed, cyclic_only=True)
            if self.compact_domains:
                generation["compact_domains"] = True
            candidates = self.generator.generate(state.domains, **generation)
            self.statistics["proposed"] += len(candidates)
            for candidate in candidates:
                if state.contradiction or state.is_solved():
                    break
                signature = candidate.core_keys, candidate.open_colors
                colors = sorted(set().union(*(self.generator.scopes[key] for key in candidate.core_keys)))
                if self.used.get(signature) == tuple(state.domains[color] for color in colors):
                    self.statistics["skipped"] += 1
                    continue
                if self.compact_domains:
                    try:
                        prepared = prepare_candidate(state, candidate, compact_domains=True, **self.limits)
                    except ValueError as error:
                        if str(error) not in ("Candidate exceeds the contraction work or tensor-size budget",
                                              "NumPy contraction supports at most 52 distinct colors"):
                            raise
                        prepared = None
                elif signature not in self.plans:
                    if len(self.plans) >= self.generator.max_steps:
                        evicted = next(iter(self.plans))
                        self.plans.pop(evicted)
                        self.used.pop(evicted, None)
                    try:
                        self.plans[signature] = prepare_candidate(state, candidate, **self.limits)
                    except ValueError as error:
                        if str(error) not in ("Candidate exceeds the contraction work or tensor-size budget",
                                              "NumPy contraction supports at most 52 distinct colors"):
                            raise
                        self.plans[signature] = None
                if not self.compact_domains:
                    prepared = self.plans[signature]
                if prepared is None:
                    self.statistics["rejected"] += 1
                    continue
                result = apply_candidate(state, candidate, prepared)
                self.statistics["applied"] += 1
                self.statistics["useful"] += int(result.removed_values > 0 or result.contradiction)
                self.statistics["removed_values"] += result.removed_values
                self.statistics["contradictions"] += int(result.contradiction)
                if not result.contradiction:
                    if self.compact_domains and len(self.used) >= self.generator.max_steps:
                        self.used.pop(next(iter(self.used)))
                    self.used[signature] = tuple(state.domains[color] for color in colors)
            self.last = None if state.contradiction else tuple(state.domains[color] for color in self.colors)
        finally:
            self.statistics["time_s"] += perf_counter() - start


def attach_heuristic(state, **options):
    """Attach cycle proposals before propagation/search, without other hooks."""
    if state.on_closure is not None:
        raise ValueError("Heuristic clusters cannot be combined with another closure hook")
    state.on_closure = HeuristicPolicy(state, **options)
    return state
