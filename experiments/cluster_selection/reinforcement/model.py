"""Construct connected tensor clusters, then learn which variables to leave open."""
from collections.abc import Mapping
from hashlib import sha1, sha256
from heapq import heapify, heappop, heappush, nsmallest
from itertools import product
import json
from math import prod
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from ..candidate_evaluation_tnreason import SummaryClusterCandidate
from ..candidate_generation import ClusterCandidateGenerator
from ..encoder import Encoder, _mlp


class Policy(Encoder):
    kind = "construction"

    def __init__(self, network, width=64, rounds=3, *, max_cores=10, max_open_colors=4,
                 max_work=1_000_000, max_size=100_000, heuristic_candidates=32, normalize=True,
                 contextual=False, context_rounds=2, exploration=.1, active_frontier=True,
                 compact_domains=False, max_total_work=None, min_active=1, cyclic_only=False,
                 hierarchical_greedy=False, target_first=False, value_encoding=False,
                 relational_context=False, relational_rounds=6, cluster_attempts=None,
                 q_learning=False, decoder="depth_first"):
        super().__init__(network, width, rounds, normalize=normalize)
        if min(max_cores, max_open_colors, max_work, max_size) < 1:
            raise ValueError("Construction limits must be positive")
        if not isinstance(heuristic_candidates, int) or isinstance(heuristic_candidates, bool) or heuristic_candidates < 0:
            raise ValueError("heuristic_candidates must be a nonnegative integer")
        self.config.update(max_cores=max_cores, max_open_colors=max_open_colors, max_work=max_work,
                           max_size=max_size, heuristic_candidates=heuristic_candidates, normalize=normalize)
        if decoder not in ("depth_first", "discrepancy", "sampling", "sampling_grow"):
            raise ValueError("Decoder must be depth_first, discrepancy, sampling or sampling_grow")
        if decoder != "depth_first":
            self.config["decoder"] = decoder
        if context_rounds < 1 or not 0 <= exploration < 1:
            raise ValueError("Expected positive context rounds and exploration in [0, 1)")
        if max_total_work is not None and max_total_work < 1:
            raise ValueError("Total contraction work must be positive")
        if cluster_attempts is not None:
            if isinstance(cluster_attempts, bool) or not isinstance(cluster_attempts, int) or cluster_attempts < 1:
                raise ValueError("Cluster attempts must be a positive integer")
            self.config["cluster_attempts"] = cluster_attempts
        if isinstance(min_active, bool) or not isinstance(min_active, int) or min_active < 1:
            raise ValueError("Minimum active variables must be a positive integer")
        if relational_context and not (contextual and value_encoding and target_first):
            raise ValueError("Relational context requires contextual, value_encoding and target_first")
        if isinstance(relational_rounds, bool) or not isinstance(relational_rounds, int) or relational_rounds < 1:
            raise ValueError("Relational rounds must be a positive integer")
        self.contextual = contextual
        self.value_encoding = value_encoding
        if not isinstance(q_learning, bool) or q_learning and not contextual:
            raise ValueError("Q-learning requires contextual encoding and a boolean q_learning flag")
        if q_learning:
            self.config["q_learning"] = True
        if contextual:
            self.config.update(contextual=True, context_rounds=context_rounds, exploration=exploration,
                               active_frontier=active_frontier, compact_domains=compact_domains, min_active=min_active)
        elif compact_domains:
            self.config["compact_domains"] = True
        if max_total_work is not None:
            self.config["max_total_work"] = max_total_work
        if cyclic_only:
            self.config["cyclic_only"] = True
        if hierarchical_greedy:
            self.config["hierarchical_greedy"] = True
        if target_first:
            self.config["target_first"] = True
        self.keys = [core["key"] for core in network["cores"]]
        self.key_idx = {key: idx for idx, key in enumerate(self.keys)}
        self.color_idx = {color: idx for idx, color in enumerate(network["colors"])}
        self.generator, self._targets_cache = None, {}
        self.scope_sets = [frozenset(scope) for scope in self.scopes]
        self.by_color = [{idx for idx, scope in enumerate(self.scope_sets) if color in scope}
                         for color in range(len(network["colors"]))]
        self.controls = nn.Embedding(2, width)
        self.head = _mlp(4 * width + (15 if contextual else 11), width, 1)
        nn.init.normal_(self.head[-1].weight, std=.001)
        nn.init.zeros_(self.head[-1].bias)
        self.beta = 3.0
        if contextual:
            # Membership alone cannot distinguish the ends and centre of a
            # selected path. Propagate on its incidence graph, then expose that
            # context to every possible extension, including nonproposal cores.
            arity = self.axes.shape[1]
            self.selection_variable = _mlp(4, width, width)
            self.selection_core = nn.Linear(1, width)
            self.selection_to_core = _mlp(2 * width + arity, width, width)
            self.selection_to_variable = _mlp(2 * width + arity, width, width)
            self.selection_update_core = _mlp(2 * width, width, width)
            self.selection_update_variable = _mlp(2 * width, width, width)
            self.extension_core = _mlp(2 * width, width, width)
            self.register_buffer("core_degree", torch.bincount(self.core_idx, minlength=len(self.keys)).float(),
                                 persistent=False)
        if value_encoding:
            self.config["value_encoding"] = True
            self._init_value_encoding()
            self.axes.zero_()  # Tuple incidences already encode tensor-axis membership.
        if relational_context:
            self.config.update(relational_context=True, relational_rounds=relational_rounds)
            self.relational_value_flags = _mlp(3, width, width)
            self.relational_tuple_flags = nn.Linear(1, width)
            self.relational_tuple_update = _mlp(2 * width + 2, width, width)

    def _init_value_encoding(self):
        """A value/allowed-tuple graph with no learned absolute-value slots."""
        for name in ("variable_encoder", "core_encoder", "to_core", "to_variable", "update_core", "update_variable"):
            delattr(self, name)
        del self.core_features
        width = self.config["width"]
        self.value_encoder = _mlp(5, width, width)
        self.value_core_encoder = _mlp(5, width, width)
        self.value_to_tuple = _mlp(2 * width, width, width)
        self.tuple_to_value = _mlp(2 * width, width, width)
        self.value_update_tuple = _mlp(2 * width, width, width)
        self.value_update_value = _mlp(2 * width, width, width)
        self.value_update_core = _mlp(2 * width, width, width)
        offsets, variable, label = [0], [], []
        for idx, dim in enumerate(self.network["dimensions"]):
            variable.extend([idx] * dim)
            label.extend(range(dim))
            offsets.append(offsets[-1] + dim)
        tuple_core, edge_tuple, edge_value, features = [], [], [], []
        for idx, core in enumerate(self.network["cores"]):
            count = sum(bool(value) for value in core["values"])
            features.append([len(core["shape"]), prod(core["shape"]), count / len(core["values"])])
            for coordinate, allowed in zip(product(*(range(dim) for dim in core["shape"])), core["values"]):
                if allowed:
                    node = len(tuple_core)
                    tuple_core.append(idx)
                    edge_tuple.extend([node] * len(coordinate))
                    edge_value.extend(offsets[color] + value for color, value in zip(core["colors"], coordinate))
        for name, values in dict(value_variable_idx=variable, value_label_idx=label,
                                 tuple_core_idx=tuple_core, tuple_edge_idx=edge_tuple,
                                 tuple_value_idx=edge_value).items():
            self.register_buffer(name, torch.tensor(values, dtype=torch.long), persistent=False)
        self.register_buffer("value_core_features", torch.tensor(features, dtype=torch.float32), persistent=False)

    def encode(self, domains):
        if not self.value_encoding:
            return super().encode(domains)
        mask = self.domain_features(domains)[:, :self.max_dim]
        cardinality = mask.sum(1)
        active = mask[self.value_variable_idx, self.value_label_idx]
        blocked = active.new_zeros(len(self.tuple_core_idx)).index_add(
            0, self.tuple_edge_idx, 1 - active[self.tuple_value_idx])
        live = (blocked == 0).nonzero().flatten()
        tuple_map = self.tuple_edge_idx.new_zeros(len(blocked))
        tuple_map[live] = torch.arange(len(live), device=mask.device)
        edges = (blocked[self.tuple_edge_idx] == 0).nonzero().flatten()
        ti = tuple_map[self.tuple_edge_idx[edges]]
        active_values = active.nonzero().flatten()
        value_map = self.tuple_value_idx.new_zeros(len(active))
        value_map[active_values] = torch.arange(len(active_values), device=mask.device)
        vi = value_map[self.tuple_value_idx[edges]]
        variables = self.value_variable_idx[active_values]
        value_degree = active.new_zeros(len(active_values)).index_add(0, vi, active.new_ones(len(vi)))
        value_features = torch.stack((active[active_values], self.dimensions[variables].log1p(),
            cardinality[variables] / self.dimensions[variables], self.degree[variables].log1p(), value_degree.log1p()), dim=1)
        values = self.value_encoder(value_features)
        tc = self.tuple_core_idx[live]
        counts = mask.new_zeros(len(self.keys)).index_add(0, tc, mask.new_ones(len(tc)))
        possible = mask.new_ones(len(self.keys)).scatter_reduce_(
            0, self.core_idx, cardinality[self.variable_idx], reduce="prod", include_self=True)
        static = self.value_core_features
        features = torch.stack((static[:, 0] / 52, static[:, 1].log1p(), static[:, 2],
                                counts / possible.clamp_min(1), counts.log1p()), dim=1)
        cores = self.value_core_encoder(features)
        tuples = cores[tc]
        tuple_degree = mask.new_zeros(len(live)).index_add(0, ti, mask.new_ones(len(ti))).clamp_min(1)
        for _ in range(self.config["rounds"]):
            messages = self.value_to_tuple(torch.cat((values[vi], tuples[ti]), dim=1))
            incoming = torch.zeros_like(tuples).index_add(0, ti, messages) / tuple_degree[:, None]
            tuples = F.layer_norm(tuples + self.value_update_tuple(torch.cat((tuples, incoming), dim=1)), tuples.shape[-1:])
            messages = self.tuple_to_value(torch.cat((tuples[ti], values[vi]), dim=1))
            incoming = torch.zeros_like(values).index_add(0, vi, messages) / value_degree.clamp_min(1)[:, None]
            values = F.layer_norm(values + self.value_update_value(torch.cat((values, incoming), dim=1)), values.shape[-1:])
        pooled = torch.zeros_like(cores).index_add(0, tc, tuples) / counts.clamp_min(1)[:, None]
        cores = F.layer_norm(cores + self.value_update_core(torch.cat((cores, pooled), dim=1)), cores.shape[-1:])
        pooled = values.new_zeros(len(self.dimensions), self.config["width"]).index_add(0, variables, values)
        encoded = pooled / cardinality.clamp_min(1)[:, None], cores
        if self.config.get("relational_context"):
            return (*encoded, dict(values=values, tuples=tuples, variables=variables,
                                   tuple_core=tc, edge_tuple=ti, edge_value=vi))
        return encoded

    def _targets(self, domains):
        """Actual structural proposals, cached by full domains, without contractions."""
        if not self.config["heuristic_candidates"]:
            return ()
        ordered = [domains[color] for color in self.network["colors"]] if isinstance(domains, Mapping) else domains
        snapshot = tuple(tuple(values) for values in ordered)
        if snapshot not in self._targets_cache:
            if self.generator is None:
                # Proposal generation needs only scopes/shapes, not copied support arrays.
                cores = {core["key"]: SimpleNamespace(colors=[self.network["colors"][idx] for idx in core["colors"]],
                                                       shape=core["shape"]) for core in self.network["cores"]}
                limits = {key: self.config[key] for key in ("max_cores", "max_open_colors", "max_work", "max_size")}
                self.generator = ClusterCandidateGenerator(cores, max_candidates=self.config["heuristic_candidates"], **limits)
            options = {"compact_domains": True} if self.config.get("compact_domains") else {}
            proposals = self.generator.generate(dict(zip(self.network["colors"], snapshot)), cyclic_only=True, **options)
            if len(self._targets_cache) >= 128:
                self._targets_cache.pop(next(iter(self._targets_cache)))
            self._targets_cache[snapshot] = tuple((frozenset(self.key_idx[key] for key in candidate.core_keys),
                                                   frozenset(self.color_idx[color] for color in candidate.open_colors))
                                                  for candidate in proposals)
        return self._targets_cache[snapshot]

    def _prior(self, actions, keys, opened, output_phase, targets):
        """Compatible complete proposals vote for their next valid construction actions."""
        selected, outputs = set(keys), set(opened)
        target_phase = self.config.get("target_first") and not selected and not outputs and not output_phase
        idx = {action: i for i, action in enumerate(actions)}
        votes = self.dimensions.new_zeros(len(actions))
        for cores, colors in targets:
            if self.config.get("target_first") and not outputs <= colors:
                continue
            if target_phase:
                following = [("open", color) for color in sorted(colors)]
            elif output_phase:
                if cores != selected or not outputs <= colors:
                    continue
                following = [("open", color) for color in sorted(colors - outputs)]
            else:
                if not selected <= cores:
                    continue
                following = [("core", key) for key in sorted(cores - selected)]
            following = [idx[action] for action in (following or [("finish", -1)]) if action in idx]
            if following:
                votes[following] += 1 / len(following)
        if not votes.sum():
            return votes  # Uniform logit offset when there is no compatible proposal.
        return (.95 * votes / votes.sum() + .05 / len(actions)).log()

    def _active_cycle_rank(self, keys, sizes=None):
        """Cycle rank of the conditioned factor incidence graph, including hyperedges."""
        sizes = self.network["dimensions"] if sizes is None else sizes
        parents = {}

        def find(color):
            parents.setdefault(color, color)
            while parents[color] != color:
                parents[color] = parents[parents[color]]
                color = parents[color]
            return color

        edges, cores = 0, 0
        for key in keys:
            scope = [color for color in self.scope_sets[key] if sizes[color] > 1]
            if not scope:
                continue
            edges += len(scope)
            cores += 1
            first = find(scope[0])
            for color in scope[1:]:
                parents[find(color)] = first
        components = len({find(color) for color in parents})
        return edges - cores - len(parents) + components

    def actions(self, keys, opened, output_phase, sizes=None):
        """Only structural/size masks; nonheuristic actions remain available."""
        target_first = self.config.get("target_first")
        if target_first and not keys and not opened and not output_phase:
            current = self.network["dimensions"] if sizes is None else sizes
            eligible = set()
            for scope in self.scope_sets:
                if len(scope) <= 52 and sum(current[color] > 1 for color in scope) >= self.config.get("min_active", 1):
                    eligible.update(scope)
            return [("skip", -1)] + [("open", color) for color, size in enumerate(current)
                if color in eligible and size > 1 and self.network["dimensions"][color] <= self.config["max_size"]]
        scope = set().union(*(self.scope_sets[key] for key in keys))
        actions = [("skip", -1)] if not self.contextual or not keys else []
        if keys and (output_phase or not self.config.get("cyclic_only") or self._active_cycle_rank(keys, sizes)):
            actions.append(("finish", -1))
        if output_phase:
            # Summary axes are expanded back to original dimensions for GAC.
            dimensions = self.network["dimensions"]
            size = prod(dimensions[color] for color in opened)
            if len(opened) < self.config["max_open_colors"]:
                actions += [("open", color) for color in sorted(scope - set(opened))
                            if size * dimensions[color] <= self.config["max_size"]
                            and (not self.contextual or sizes is None or sizes[color] > 1)]
        elif len(keys) < self.config["max_cores"]:
            bridges = {color for color in scope if sizes is None or not self.contextual
                       or not self.config["active_frontier"] or sizes[color] > 1}
            if keys:
                frontier = set().union(*(self.by_color[color] for color in bridges))
            elif target_first:
                frontier = self.by_color[opened[0]]
            else:
                frontier = set(range(len(self.keys)))
            actions += [("core", key) for key in sorted(frontier - set(keys))
                        if len(scope | self.scope_sets[key]) <= 52
                        and (not self.contextual or sizes is None or sum(sizes[c] > 1 for c in self.scope_sets[key])
                             >= self.config["min_active"])]
        return actions or [("skip", -1)]

    def distribution(self, encoded, sizes, keys, opened, output_phase, beta, targets=()):
        if self.contextual:
            return self._context_distribution(encoded, sizes, keys, opened, output_phase, beta, targets)
        return self._legacy_distribution(encoded, sizes, keys, opened, output_phase, beta, targets)

    def q_values(self, encoded, sizes, keys, opened, output_phase):
        """Raw action values over the same legal construction choices as inference."""
        if not self.contextual:
            raise ValueError("Q values require contextual encoding")
        return self._context_distribution(encoded, sizes, keys, opened, output_phase, 0., (),
                                          return_values=True)

    def _legacy_distribution(self, encoded, sizes, keys, opened, output_phase, beta, targets):
        variables, cores = encoded
        scope = sorted(set().union(*(self.scope_sets[key] for key in keys)))
        active = {color for color in scope if sizes[color] > 1}
        cycles = sum(len(self.scopes[key]) for key in keys) - len(keys) - len(scope) + int(bool(keys))
        zero = variables.new_zeros(self.config["width"])
        context = torch.cat((cores[keys].mean(0) if keys else zero,
                             variables[scope].mean(0) if scope else zero,
                             variables[opened].mean(0) if opened else zero))
        conditioned = self.config.get("cyclic_only") or self.config.get("target_first")
        actions = self.actions(keys, opened, output_phase, sizes if conditioned else None)
        embeddings, features = [], []
        for kind, idx in actions:
            pressure, overlap = 0.0, 0
            if kind == "core":
                selected = self.scope_sets[idx]
                pressure = sum(1 - sizes[c] / self.network["dimensions"][c] for c in selected if sizes[c] > 1) / max(1, len(selected))
                overlap = len(selected & active)
                embedding = cores[idx]
            elif kind == "open":
                dim = self.network["dimensions"][idx]
                pressure = 1 - sizes[idx] / dim
                embedding = variables[idx]
            else:
                embedding = self.controls.weight[int(kind == "skip")]
            embeddings.append(torch.cat((context, embedding)))
            features.append([float(output_phase), len(keys) / self.config["max_cores"],
                             len(opened) / self.config["max_open_colors"], cycles / self.config["max_cores"],
                             len(scope) / 52, pressure, min(overlap, 2) / 2,
                             *[float(kind == label) for label in ("core", "open", "finish", "skip")]])
        residual = self.head(torch.cat((torch.stack(embeddings), variables.new_tensor(features)), dim=1)).squeeze(1)
        prior = self._prior(actions, keys, opened, output_phase, targets) if beta else 0
        return actions, Categorical(logits=residual + beta * prior)

    def _selection_graph(self, keys):
        selected = self.dimensions.new_zeros(len(self.keys))
        selected[keys] = 1
        edge_mask = selected[self.core_idx].bool()
        vi, ci, axes = self.variable_idx[edge_mask], self.core_idx[edge_mask], self.axes[edge_mask]
        degree = torch.bincount(vi, minlength=len(self.dimensions)).to(self.dimensions.dtype)
        neighbor_degree = torch.zeros_like(degree)
        if len(vi):
            sums = self.dimensions.new_zeros(len(self.keys)).index_add(0, ci, degree[vi])
            neighbors = self.dimensions.new_zeros(len(degree)).index_add(0, vi, self.core_degree[ci] - 1)
            neighbor_degree = neighbor_degree.index_add(0, vi, sums[ci] - degree[vi]) / neighbors.clamp_min(1)
        return vi, ci, axes, degree, neighbor_degree

    def selected_encoding(self, encoded, keys, opened, candidates=None, graph=None):
        """Expose partial-cluster topology without rerunning the full encoder."""
        variables, cores = encoded[:2]
        if self.config.get("relational_context") and opened:
            return self._relational_encoding(encoded, keys, opened[0], candidates)
        vi, ci, axes, degree, neighbor_degree = self._selection_graph(keys) if graph is None else graph
        if len(vi):
            scope, local_vi = torch.unique(vi, sorted=True, return_inverse=True)
            selected_keys = torch.tensor(keys, device=cores.device)
            key_map = self.core_idx.new_zeros(len(cores))
            key_map[selected_keys] = torch.arange(len(keys), device=cores.device)
            local_ci = key_map[ci]
            outputs = variables.new_zeros(len(variables))
            outputs[opened] = 1
            flags = torch.stack((torch.ones_like(degree[scope]), outputs[scope],
                                 degree[scope].log1p(), neighbor_degree[scope].log1p()), dim=1)
            local_variables = F.layer_norm(variables[scope] + self.selection_variable(flags), variables.shape[-1:])
            local_cores = F.layer_norm(cores[selected_keys] + self.selection_core(
                cores.new_ones(len(keys), 1)), cores.shape[-1:])
            for _ in range(self.config["context_rounds"]):
                messages = self.selection_to_core(torch.cat((local_variables[local_vi], local_cores[local_ci], axes), dim=1))
                incoming = torch.zeros_like(local_cores).index_add(0, local_ci, messages)
                incoming = incoming / self.core_degree[selected_keys].clamp_min(1)[:, None]
                local_cores = F.layer_norm(local_cores + self.selection_update_core(
                    torch.cat((local_cores, incoming), dim=1)), cores.shape[-1:])
                messages = self.selection_to_variable(torch.cat((local_cores[local_ci], local_variables[local_vi], axes), dim=1))
                incoming = torch.zeros_like(local_variables).index_add(0, local_vi, messages) / degree[scope, None]
                local_variables = F.layer_norm(local_variables + self.selection_update_variable(
                    torch.cat((local_variables, incoming), dim=1)), variables.shape[-1:])
            variables = variables.index_copy(0, scope, local_variables)
            cores = cores.index_copy(0, selected_keys, local_cores)
        elif opened and self.config.get("target_first"):
            # Before any core is chosen, expose the output goal to the seed
            # scorer and its incident endpoints as well as the global context.
            targets = torch.tensor(opened, device=variables.device)
            flags = variables.new_zeros(len(opened), 4)
            flags[:, 1] = 1
            marked = F.layer_norm(variables[targets] + self.selection_variable(flags), variables.shape[-1:])
            variables = variables.index_copy(0, targets, marked)
        endpoints = torch.zeros_like(cores).index_add(0, self.core_idx, variables[self.variable_idx])
        endpoints = endpoints / self.core_degree.clamp_min(1)[:, None]
        if candidates is None:
            candidates = torch.arange(len(cores), device=cores.device)
        else:
            candidates = torch.tensor(candidates, device=cores.device, dtype=torch.long)
        extensions = cores[candidates]
        extensions = F.layer_norm(extensions + self.extension_core(
            torch.cat((extensions, endpoints[candidates]), dim=1)), cores.shape[-1:])
        cores = cores.index_copy(0, candidates, extensions)
        return variables, cores

    def _relational_encoding(self, encoded, keys, target, candidates):
        """Preserve value correspondence while querying the selected relations.

        One query anchors each allowed target value. Averaging queries removes
        the anchor's label, while nonlinear tuple pooling retains whether two
        endpoints carry the same query through the selected path.
        """
        variables, cores, data = encoded
        base_values, base_tuples = data["values"], data["tuples"]
        vv, tc, ti, vi = (data[key] for key in ("variables", "tuple_core", "edge_tuple", "edge_value"))
        selected = torch.zeros(len(cores), dtype=torch.bool, device=cores.device)
        selected[keys] = True
        scope = torch.zeros(len(variables), dtype=torch.bool, device=cores.device)
        scope[self.variable_idx[selected[self.core_idx]]] = True
        local = (scope[vv] | (vv == target)).nonzero().flatten()
        value_map = torch.zeros(len(base_values), dtype=torch.long, device=cores.device)
        value_map[local] = torch.arange(len(local), device=cores.device)
        anchors = value_map[(vv == target).nonzero().flatten()]
        queries, width = len(anchors), self.config["width"]
        flags = base_values.new_zeros(queries, len(local), 3)
        flags[:, :, 0] = scope[vv[local]]
        flags[:, :, 1] = vv[local] == target
        flags[torch.arange(queries, device=cores.device), anchors, 2] = 1
        values = F.layer_norm(base_values[local][None] + self.relational_value_flags(flags), (width,))
        signal = flags[:, :, 2]
        selected_tuples = selected[tc].nonzero().flatten()
        if len(selected_tuples):
            tuple_map = torch.zeros(len(tc), dtype=torch.long, device=cores.device)
            tuple_map[selected_tuples] = torch.arange(len(selected_tuples), device=cores.device)
            edge = selected[tc[ti]]
            sti, svi = tuple_map[ti[edge]], value_map[vi[edge]]
            tuple_degree = torch.bincount(sti, minlength=len(selected_tuples)).clamp_min(1)
            value_degree = torch.bincount(svi, minlength=len(local)).clamp_min(1)
            tuples = F.layer_norm(base_tuples[selected_tuples][None] + self.relational_tuple_flags(
                base_tuples.new_ones(1, len(selected_tuples), 1)), (width,)).expand(queries, -1, -1)
            initial_signal = signal
            for _ in range(self.config["relational_rounds"]):
                # Keep the anchor channel at the target boundary and diffuse
                # it through selected tuple incidences. This prevents learned
                # message attenuation from erasing a multi-edge query.
                tuple_signal = signal.new_zeros(queries, len(selected_tuples)).index_add(
                    1, sti, signal[:, svi]) / tuple_degree[None]
                signal = signal.new_zeros(queries, len(local)).index_add(
                    1, svi, tuple_signal[:, sti]) / value_degree[None]
                signal = initial_signal + signal * (vv[local] != target)[None]
                query_flags = torch.stack((flags[:, :, 0], flags[:, :, 1], signal), dim=-1)
                messages = self.value_to_tuple(torch.cat((values[:, svi], tuples[:, sti]), dim=-1))
                incoming = torch.zeros_like(tuples).index_add(1, sti, messages) / tuple_degree[None, :, None]
                joint = torch.cat((tuples, incoming, tuple_signal[:, :, None], tuple_signal.square()[:, :, None]), dim=-1)
                tuples = F.layer_norm(tuples + self.relational_tuple_update(joint), (width,))
                messages = self.tuple_to_value(torch.cat((tuples[:, sti], values[:, svi]), dim=-1))
                incoming = torch.zeros_like(values).index_add(1, svi, messages) / value_degree[None, :, None]
                values = F.layer_norm(values + self.relational_value_flags(query_flags)
                    + self.value_update_value(torch.cat((values, incoming), dim=-1)), (width,))
        # Pool the updated endpoints through each possible extension's actual
        # allowed tuples, before pooling values or queries. No contraction is
        # performed here; domains still condition the live tuple graph.
        if candidates is None:
            candidates = torch.arange(len(cores), device=cores.device)
        else:
            candidates = torch.tensor(candidates, device=cores.device, dtype=torch.long)
        candidate_mask = torch.zeros_like(selected)
        candidate_mask[candidates] = True
        ct = candidate_mask[tc].nonzero().flatten()
        candidate_map = torch.zeros(len(cores), dtype=torch.long, device=cores.device)
        candidate_map[candidates] = torch.arange(len(candidates), device=cores.device)
        if len(ct):
            tuple_map = torch.zeros(len(tc), dtype=torch.long, device=cores.device)
            tuple_map[ct] = torch.arange(len(ct), device=cores.device)
            edge = candidate_mask[tc[ti]]
            eti, evi = tuple_map[ti[edge]], vi[edge]
            all_values = base_values[None].expand(queries, -1, -1).index_copy(1, local, values)
            tuples = F.layer_norm(base_tuples[ct][None] + self.relational_tuple_flags(
                selected[tc[ct]].to(cores.dtype)[None, :, None]), (width,)).expand(queries, -1, -1)
            messages = self.value_to_tuple(torch.cat((all_values[:, evi], tuples[:, eti]), dim=-1))
            incoming = torch.zeros_like(tuples).index_add(1, eti, messages)
            arity = torch.bincount(eti, minlength=len(ct)).clamp_min(1)
            incoming = incoming / arity[None, :, None]
            all_signal = signal.new_zeros(queries, len(base_values)).index_copy(1, local, signal)
            tuple_signal = signal.new_zeros(queries, len(ct)).index_add(1, eti, all_signal[:, evi]) / arity[None]
            # The square is taken before tuple/query pooling: it contains
            # cross-endpoint correlation that pooled first moments discard.
            joint = torch.cat((tuples, incoming, tuple_signal[:, :, None], tuple_signal.square()[:, :, None]), dim=-1)
            tuples = F.layer_norm(tuples + self.relational_tuple_update(joint), (width,))
            ci = candidate_map[tc[ct]]
            pooled = tuples.new_zeros(queries, len(candidates), width).index_add(1, ci, tuples)
            pooled = (pooled / torch.bincount(ci, minlength=len(candidates)).clamp_min(1)[None, :, None]).mean(0)
            extensions = cores[candidates]
            extensions = F.layer_norm(extensions + self.value_update_core(
                torch.cat((extensions, pooled), dim=-1)), (width,))
            cores = cores.index_copy(0, candidates, extensions)
        local_variables = vv[local]
        pooled = values.new_zeros(len(variables), width).index_add(0, local_variables, values.mean(0))
        cardinality = torch.bincount(local_variables, minlength=len(variables))
        changed = cardinality.nonzero().flatten()
        variables = variables.index_copy(0, changed, pooled[changed] / cardinality[changed, None])
        anchor_mean = values[torch.arange(queries, device=cores.device), anchors].mean(0)
        return variables.index_copy(0, vv.new_tensor([target]), anchor_mean[None]), cores

    def _context_distribution(self, encoded, sizes, keys, opened, output_phase, beta, targets,
                              *, return_values=False):
        scope = sorted(set().union(*(self.scope_sets[key] for key in keys)))
        actions = self.actions(keys, opened, output_phase, sizes)
        graph = self._selection_graph(keys)
        candidates = sorted(set(keys) | {idx for kind, idx in actions if kind == "core"})
        variables, cores = self.selected_encoding(encoded, keys, opened, candidates, graph)
        zero = variables.new_zeros(self.config["width"])
        context = torch.cat((cores[keys].mean(0) if keys else zero,
                             variables[scope].mean(0) if scope else zero,
                             variables[opened].mean(0) if opened else zero))
        # One batched gather and head call, rather than one tensor construction
        # per core. Core pressure/overlap are reductions on cached incidence.
        labels = {"core": 0, "open": 1, "finish": 2, "skip": 3}
        kinds = torch.tensor([labels[kind] for kind, _ in actions], device=variables.device)
        offsets = {"core": 0, "open": len(cores), "finish": len(cores) + len(variables),
                   "skip": len(cores) + len(variables) + 1}
        indices = torch.tensor([offsets[kind] + (idx if idx >= 0 else 0) for kind, idx in actions],
                               device=variables.device)
        embeddings = torch.cat((cores, variables, self.controls.weight))[indices]
        domain_sizes = variables.new_tensor(sizes)
        pressure = 1 - domain_sizes / self.dimensions
        active = variables.new_zeros(len(variables))
        active[scope] = (domain_sizes[scope] > 1).to(variables.dtype)
        core_pressure = variables.new_zeros(len(cores)).index_add(
            0, self.core_idx, (pressure * (domain_sizes > 1))[self.variable_idx]) / self.core_degree.clamp_min(1)
        overlap = variables.new_zeros(len(cores)).index_add(0, self.core_idx, active[self.variable_idx])
        all_pressure = torch.cat((core_pressure, pressure, variables.new_zeros(2)))
        all_overlap = torch.cat((overlap.clamp_max(2) / 2, variables.new_zeros(len(variables) + 2)))
        cycles = sum(len(self.scopes[key]) for key in keys) - len(keys) - len(scope) + int(bool(keys))
        common = variables.new_tensor([float(output_phase), len(keys) / self.config["max_cores"],
                                       len(opened) / self.config["max_open_colors"],
                                       cycles / self.config["max_cores"], len(scope) / 52])
        degree, neighbor_degree = graph[-2:]
        node_topology = torch.stack((degree.log1p(), neighbor_degree.log1p(),
                                    (degree == 1).to(variables.dtype), (degree > 0).to(variables.dtype)), dim=1)
        core_topology = cores.new_zeros(len(cores), 4).index_add(0, self.core_idx, node_topology[self.variable_idx])
        core_topology = core_topology / self.core_degree.clamp_min(1)[:, None]
        topology = torch.cat((core_topology, node_topology, variables.new_zeros(2, 4)))
        features = torch.cat((common.expand(len(actions), -1), all_pressure[indices, None],
                              all_overlap[indices, None], F.one_hot(kinds, 4).to(variables.dtype), topology[indices]), dim=1)
        residual = self.head(torch.cat((context.expand(len(actions), -1), embeddings, features), dim=1)).squeeze(1)
        if return_values:
            return actions, residual
        if self.config.get("q_learning"):
            return actions, Categorical(logits=residual)
        # A trainable scorer starts with a mild preference for completing a
        # nonempty construction. Guidance is a mixture, so beta cannot suppress
        # every novel extension exponentially as in legacy checkpoints.
        residual = residual - 2 * (kinds == 3) - 1 * ((kinds == 2) & (not output_phase))
        probabilities = residual.softmax(0)
        if beta and targets:
            prior = self._prior(actions, keys, opened, output_phase, targets)
            if prior.any():
                weight = min(.75, beta / (1 + beta))
                probabilities = (1 - weight) * probabilities + weight * prior.softmax(0)
        epsilon = self.config["exploration"] if self.training else 0.
        if epsilon:
            probabilities = (1 - epsilon) * probabilities + epsilon / len(actions)
        return actions, Categorical(probs=probabilities)

    def candidate_trace(self, candidate, domains=None, *, rng=None, target_color=None):
        """Order a measured teacher legally; random orders cover useful prefixes."""
        remaining = {self.key_idx[key] for key in candidate.core_keys}
        if not remaining:
            raise ValueError("A candidate must contain at least one core")
        sizes = None
        if domains is not None:
            ordered = [domains[color] for color in self.network["colors"]] if isinstance(domains, Mapping) else domains
            sizes = [len(set(values)) for values in ordered]
        keys, trace, opened = [], [], []
        outputs = [self.color_idx[color] for color in candidate.open_colors]
        if self.config.get("target_first"):
            legal = {idx for kind, idx in self.actions([], [], False, sizes) if kind == "open"}
            eligible = [idx for idx in outputs if idx in legal]
            if not eligible:
                raise ValueError("Target-first construction requires an active, affordable teacher output")
            if target_color is not None:
                target = self.color_idx.get(target_color)
                if target not in eligible:
                    raise ValueError("Requested target must be an active, affordable teacher output")
            else:
                preferred = [idx for idx in eligible if self.network["colors"][idx] in (self.reward_colors or ())]
                target = (preferred or eligible)[0] if rng is None else rng.choice(eligible)
            opened.append(target)
            outputs.remove(target)
            trace.append(("open", target))
        elif target_color is not None:
            raise ValueError("Explicit target requires target_first construction")
        while remaining:
            legal = {idx for kind, idx in self.actions(keys, opened, False, sizes) if kind == "core"}
            following = remaining & legal
            if not following:
                raise ValueError("Candidate exceeds construction limits or is disconnected")
            idx = min(following) if rng is None else rng.choice(sorted(following))
            keys.append(idx)
            remaining.remove(idx)
            trace.append(("core", idx))
        if ("finish", -1) not in self.actions(keys, opened, False, sizes):
            raise ValueError("Candidate cannot finish under the construction masks")
        trace.append(("finish", -1))
        if rng is not None:
            rng.shuffle(outputs)
        for color in outputs:
            if ("open", color) not in self.actions(keys, opened, True, sizes):
                raise ValueError("Teacher outputs exceed the construction masks")
            opened.append(color)
            trace.append(("open", color))
        trace.append(("finish", -1))
        return trace

    def _greedy_choice(self, actions, distribution):
        if self.config.get("q_learning") or not self.config.get("hierarchical_greedy"):
            return distribution.logits.argmax()
        # Stopping competes with the total continuation probability. Comparing
        # it to each interchangeable next core separately can stop too early.
        labels = {"core": 0, "open": 1, "finish": 2, "skip": 3}
        kinds = torch.tensor([labels[kind] for kind, _ in actions], device=distribution.probs.device)
        masses = distribution.probs.new_zeros(4).index_add(0, kinds, distribution.probs)
        return distribution.probs.masked_fill(kinds != masses.argmax(), -1).argmax()

    def _continuations(self, actions, keys, opened, output_phase):
        """Growth keeps legal extensions, then finish, then original fallbacks."""
        target = self.config.get("target_first") and not keys and not opened and not output_phase
        kind = "open" if output_phase or target else "core"
        following = [idx for idx, action in enumerate(actions) if action[0] == kind]
        return (following or [idx for idx, action in enumerate(actions) if action[0] == "finish"]
                or list(range(len(actions))))

    def rollout(self, domains, *, beta=None, greedy=False, trace=None, prefix=None):
        """One state encoding; bounded growth followed by output selection.

        Return candidate (or skip), summed log probability, entropy, and actions.
        A full trace scores a teacher without sampling; a forced prefix is
        completed and only its sampled suffix receives policy-gradient credit.
        Training samples; inference chooses argmax. Empty outputs are legal.
        Growth decoding filters free actions; forced teachers/prefixes bypass it.
        """
        if trace is not None and prefix is not None:
            raise ValueError("Specify either a complete trace or a prefix")
        forced = list(trace if trace is not None else prefix or ())
        complete_trace = trace is not None
        beta = 0. if self.config.get("q_learning") else self.beta if beta is None else beta
        encoded = self.encode(domains)
        targets = self._targets(domains) if beta else ()
        ordered = [domains[color] for color in self.network["colors"]] if isinstance(domains, Mapping) else domains
        sizes = [len(set(values)) for values in ordered]
        keys, opened, output_phase, decisions, log_probs, entropies = [], [], False, [], [], []
        while True:
            actions, distribution = self.distribution(encoded, sizes, keys, opened, output_phase, beta, targets)
            if (self.config.get("decoder") == "sampling_grow" and not complete_trace
                    and len(decisions) >= len(forced)):
                following = self._continuations(actions, keys, opened, output_phase)
                actions = [actions[idx] for idx in following]
                distribution = Categorical(logits=distribution.logits[following])
            if len(decisions) < len(forced):
                action = tuple(forced[len(decisions)])
                if action not in actions:
                    raise ValueError(f"Illegal construction action {action}")
                choice = distribution.logits.new_tensor(actions.index(action), dtype=torch.long)
            elif complete_trace:
                raise ValueError("Construction trace ended before completion")
            else:
                choice = self._greedy_choice(actions, distribution) if greedy else distribution.sample()
            action = actions[int(choice)]
            scored = complete_trace or len(decisions) >= len(forced)
            decisions.append(action)
            if scored:
                log_probs.append(distribution.log_prob(choice))
                entropies.append(distribution.entropy())
            kind, idx = action
            if kind == "core":
                keys.append(idx)
            elif kind == "open":
                opened.append(idx)
            elif kind == "finish" and not output_phase:
                output_phase = True
            else:
                break
        if len(decisions) < len(forced):
            raise ValueError("Construction trace contains actions after completion")
        candidate = None
        if kind == "finish":
            core_keys = tuple(self.keys[idx] for idx in sorted(keys))
            colors = tuple(self.network["colors"][idx] for idx in sorted(opened))
            name = "rl_" + sha1(repr((core_keys, colors)).encode()).hexdigest()
            candidate = SummaryClusterCandidate(name, core_keys, colors)
        log_prob = torch.stack(log_probs).sum() if log_probs else self.head[-1].weight.sum() * 0
        entropy = torch.stack(entropies).mean() if entropies else log_prob
        return candidate, log_prob, entropy, decisions

    @torch.no_grad()
    def propose(self, domains):
        return self.rollout(domains, greedy=True)[0]

    def proposals(self, domains, *, max_candidates=32, max_steps=8192):
        """Enumerate distinct policy-ranked constructions, with bounded search.

        The first branch follows greedy inference. Default depth-first search
        backtracks through the same action space. Discrepancy search orders paths
        by summed local action ranks, favouring earlier deviations on ties. Its
        frontier holds at most 2 * max_steps entries; at capacity it retains the
        best max_steps. Neither mode compares Q values between different states.
        Sampling follows the first greedy path, then draws complete paths with a
        domain-seeded local generator. Duplicate/skip paths consume scored states;
        only the current path is retained. sampling_grow filters legal actions to
        extensions, then finish, then fallback actions. Runtime still checks work
        affordability. Q softmax sampling is experimental.
        Restart the iterator when domains change.
        """
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (max_candidates, max_steps)):
            raise ValueError("Proposal and construction-step limits must be positive integers")
        with torch.no_grad():
            encoded = self.encode(domains)
            targets = self._targets(domains) if self.beta and not self.config.get("q_learning") else ()
        ordered = [domains[color] for color in self.network["colors"]] if isinstance(domains, Mapping) else domains
        sizes = [len(set(values)) for values in ordered]
        discrepancy = self.config.get("decoder", "depth_first") == "discrepancy"
        sampling = self.config.get("decoder") in ("sampling", "sampling_grow")
        grow = self.config.get("decoder") == "sampling_grow"
        first_path = True
        if sampling:
            canonical = [(color, sorted(set(values))) for color, values in zip(self.network["colors"], ordered)]
            seed = int(sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest(), 16) % (2 ** 63)
            generator = torch.Generator(device=next(self.parameters()).device).manual_seed(seed)
        stack = [(0, float("inf"), 0, (), (), 0, 0)] if discrepancy else [((), (), 0)]
        visited, yielded, serial = set(), set(), 0
        steps = 0
        while stack:
            if discrepancy:
                distance, deviation, _, keys, opened, phase, depth = heappop(stack)
            else:
                keys, opened, phase = stack.pop()
            if phase == 2:
                core_keys = tuple(self.keys[idx] for idx in sorted(keys))
                colors = tuple(self.network["colors"][idx] for idx in sorted(opened))
                signature = core_keys, colors
                if sampling:
                    first_path = False
                    stack.append(((), (), 0))
                if signature in yielded:
                    continue
                yielded.add(signature)
                name = "rl_" + sha1(repr(signature).encode()).hexdigest()
                # No no_grad context crosses a yield into the caller.
                yield SummaryClusterCandidate(name, core_keys, colors)
                if len(yielded) >= max_candidates:
                    return
                continue
            anchor = opened[0] if opened and self.config.get("target_first") else None
            state = frozenset(keys), frozenset(opened), phase, anchor
            if not sampling and state in visited:
                continue
            if steps >= max_steps:
                return
            if not sampling:
                visited.add(state)
            steps += 1
            with torch.no_grad():
                q_learning = self.config.get("q_learning")
                if q_learning:
                    actions, values = self.q_values(encoded, sizes, list(keys), list(opened), bool(phase))
                else:
                    actions, distribution = self.distribution(
                        encoded, sizes, list(keys), list(opened), bool(phase), self.beta, targets)
                if grow:
                    following = self._continuations(actions, keys, opened, bool(phase))
                    actions = [actions[idx] for idx in following]
                    if q_learning:
                        values = values[following]
                    else:
                        distribution = Categorical(logits=distribution.logits[following])
                if q_learning:
                    if sampling:
                        choice = values.argmax() if first_path else torch.multinomial(
                            values.softmax(0), 1, generator=generator).squeeze(0)
                    else:
                        first, scores = int(values.argmax()), values.tolist()
                else:
                    if sampling:
                        choice = self._greedy_choice(actions, distribution) if first_path else torch.multinomial(
                            distribution.probs, 1, generator=generator).squeeze(0)
                    else:
                        first, scores = int(self._greedy_choice(actions, distribution)), distribution.probs.tolist()
            if sampling:
                ranked = [int(choice)]
            else:
                ranked = [first] + sorted((idx for idx in range(len(actions)) if idx != first),
                                          key=lambda idx: -scores[idx])
            for rank in reversed(range(len(ranked))):
                choice = ranked[rank]
                kind, idx = actions[choice]
                if kind == "core":
                    following = keys + (idx,), opened, phase
                elif kind == "open":
                    following = keys, opened + (idx,), phase
                elif kind == "finish":
                    following = keys, opened, phase + 1
                else:
                    if sampling:
                        first_path = False
                        stack.append(((), (), 0))
                    continue
                if discrepancy:
                    if len(stack) >= 2 * max_steps:
                        stack[:] = nsmallest(max_steps, stack)
                        heapify(stack)
                    serial += 1
                    heappush(stack, (distance + rank, min(deviation, depth) if rank else deviation,
                                     serial, *following, depth + 1))
                else:
                    stack.append(following)


def load(path):
    from ..runtime import load as load_model
    model = load_model(path)
    if getattr(model, "kind", None) != Policy.kind:
        raise ValueError("Expected a version-1 construction-policy checkpoint")
    return model
