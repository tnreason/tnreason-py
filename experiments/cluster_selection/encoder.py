"""Domain-conditioned variable/core encoder shared by both learning paradigms."""
from collections.abc import Mapping
from math import prod
from numbers import Integral
import torch
from torch import nn
from torch.nn import functional as F


def _mlp(input_size, width, output_size):
    return nn.Sequential(nn.Linear(input_size, width), nn.ReLU(), nn.Linear(width, output_size))


class Encoder(nn.Module):
    def __init__(self, network, width=64, rounds=3, *, normalize=False):
        super().__init__()
        if width < 1 or rounds < 1 or not network["colors"] or not network["cores"]:
            raise ValueError("Expected a nonempty network and positive width/rounds")
        if normalize and width < 2:
            raise ValueError("Normalized encoders require width >= 2")
        self.network, self.config = network, dict(width=width, rounds=rounds)
        self.normalize = normalize
        self.reward_colors = None
        self.scopes = [core["colors"] for core in network["cores"]]
        self.max_dim = max(network["dimensions"])
        arity = max(1, max(map(len, self.scopes)))
        entries = max(len(core["values"]) for core in network["cores"])
        dimensions = torch.tensor(network["dimensions"], dtype=torch.float32)
        valid = torch.arange(self.max_dim)[None, :] < dimensions[:, None]
        features = torch.zeros(len(self.scopes), entries + arity)
        edges, self.offsets = [], [0]
        for idx, core in enumerate(network["cores"]):
            if len(core["shape"]) != len(core["colors"]) or prod(core["shape"]) != len(core["values"]):
                raise ValueError("Core support must match its ordered axes and shape")
            if any(network["dimensions"][color] != dim for color, dim in zip(core["colors"], core["shape"])):
                raise ValueError("Core axes must match variable dimensions")
            features[idx, :len(core["values"])] = torch.tensor(core["values"], dtype=torch.float32)
            features[idx, entries:entries + len(core["shape"])] = torch.tensor(core["shape"]) / self.max_dim
            edges.extend((color, idx, axis) for axis, color in enumerate(core["colors"]))
            self.offsets.append(len(edges))
        edge = torch.tensor(edges, dtype=torch.long).reshape(-1, 3)
        degree = torch.bincount(edge[:, 0], minlength=len(dimensions)).float()
        for name, value in dict(dimensions=dimensions, valid=valid, core_features=features,
                                variable_idx=edge[:, 0], core_idx=edge[:, 1],
                                axes=torch.eye(arity)[edge[:, 2]], degree=degree).items():
            self.register_buffer(name, value, persistent=False)
        self.variable_encoder = _mlp(2 * self.max_dim + 4, width, width)
        self.core_encoder = _mlp(entries + arity, width, width)
        self.to_core = _mlp(2 * width + arity, width, width)
        self.to_variable = _mlp(2 * width + arity, width, width)
        self.update_core = _mlp(2 * width, width, width)
        self.update_variable = _mlp(2 * width, width, width)

    def domain_features(self, domains):
        if isinstance(domains, Mapping):
            domains = [domains[color] for color in self.network["colors"]]
        if len(domains) != len(self.dimensions):
            raise ValueError("Expected every variable's domain in network order")
        mask = self.dimensions.new_zeros(len(domains), self.max_dim)
        for idx, values in enumerate(domains):
            if any(isinstance(value, bool) or not isinstance(value, Integral)
                   or not 0 <= value < self.network["dimensions"][idx] for value in values):
                raise ValueError("Domain value outside variable dimension")
            mask[idx, list(values)] = 1
        scalars = torch.stack((self.dimensions / self.max_dim, mask.sum(1) / self.dimensions,
                               (self.dimensions == 2).to(mask.dtype), self.degree.log1p()), dim=1)
        return torch.cat((mask, self.valid.to(mask.dtype), scalars), dim=1)

    def _update(self, nodes, messages, idx, layer):
        incoming = torch.zeros_like(nodes).index_add(0, idx, messages)
        if self.normalize:
            incoming = incoming / torch.bincount(idx, minlength=len(nodes)).clamp_min(1)[:, None]
        nodes = nodes + layer(torch.cat((nodes, incoming), dim=1))
        return F.layer_norm(nodes, nodes.shape[-1:]) if self.normalize else nodes

    def _round(self, variables, cores, vi, ci, axes):
        messages = self.to_core(torch.cat((variables[vi], cores[ci], axes), dim=1))
        cores = self._update(cores, messages, ci, self.update_core)
        messages = self.to_variable(torch.cat((cores[ci], variables[vi], axes), dim=1))
        variables = self._update(variables, messages, vi, self.update_variable)
        return variables, cores

    def encode(self, domains):
        variables = self.variable_encoder(self.domain_features(domains))
        cores = self.core_encoder(self.core_features)
        for _ in range(self.config["rounds"]):
            variables, cores = self._round(variables, cores, self.variable_idx, self.core_idx, self.axes)
        return variables, cores
