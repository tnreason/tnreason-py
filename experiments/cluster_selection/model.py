"""Shared state encoder and candidate-local scorer; no solver dependencies."""
from collections.abc import Mapping
from math import prod
from numbers import Integral
import torch
from torch import nn


def _mlp(input_size, width, output_size):
    return nn.Sequential(nn.Linear(input_size, width), nn.ReLU(), nn.Linear(width, output_size))


class Predictor(nn.Module):
    """Predict log1p(gain), log(seconds), and contradiction logit, in that order.

    The ordered raw network fixes the schema. Domains may be ordered lists or a
    mapping keyed by color; candidates use core/color indices and shape estimates.
    Weights are shared across rounds. Tensor axes are never silently reordered.
    """

    def __init__(self, network, width=64, rounds=3):
        super().__init__()
        if width < 1 or rounds < 1 or not network["colors"] or not network["cores"]:
            raise ValueError("Expected a nonempty network and positive width/rounds")
        self.network, self.config = network, dict(width=width, rounds=rounds)
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
        self.roles = nn.Embedding(2, width)
        self.head = nn.Sequential(nn.Linear(3 * width + 6, width), nn.ReLU(),
                                  nn.Linear(width, 32), nn.ReLU(), nn.Linear(32, 3))

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

    def _round(self, variables, cores, vi, ci, axes):
        messages = self.to_core(torch.cat((variables[vi], cores[ci], axes), dim=1))
        cores = cores + self.update_core(torch.cat((cores, torch.zeros_like(cores).index_add(0, ci, messages)), dim=1))
        messages = self.to_variable(torch.cat((cores[ci], variables[vi], axes), dim=1))
        variables = variables + self.update_variable(torch.cat(
            (variables, torch.zeros_like(variables).index_add(0, vi, messages)), dim=1))
        return variables, cores

    def encode(self, domains):
        variables = self.variable_encoder(self.domain_features(domains))
        cores = self.core_encoder(self.core_features)
        for _ in range(self.config["rounds"]):
            variables, cores = self._round(variables, cores, self.variable_idx, self.core_idx, self.axes)
        return variables, cores

    def forward(self, domains, candidates):
        if not candidates:
            return self.dimensions.new_empty((0, 3))
        variables, cores = self.encode(domains)
        predictions = []
        for candidate in candidates:
            keys, opened = candidate["cores"], candidate["open"]
            if not keys or len(set(keys)) != len(keys) or any(key < 0 or key >= len(self.scopes) for key in keys):
                raise ValueError("Expected distinct valid candidate cores")
            scope = sorted({color for key in keys for color in self.scopes[key]})
            if len(set(opened)) != len(opened) or not set(opened) <= set(scope):
                raise ValueError("Candidate outputs must be distinct variables in its cores")
            idx = {color: i for i, color in enumerate(scope)}
            vi = [idx[color] for key in keys for color in self.scopes[key]]
            ci = [i for i, key in enumerate(keys) for _ in self.scopes[key]]
            edge = [i for key in keys for i in range(self.offsets[key], self.offsets[key + 1])]
            device = self.dimensions.device
            vi, ci = torch.tensor(vi, dtype=torch.long, device=device), torch.tensor(ci, dtype=torch.long, device=device)
            role = torch.tensor([color in opened for color in scope], dtype=torch.long, device=device)
            local_v, local_c = self._round(variables[scope] + self.roles(role), cores[keys], vi, ci, self.axes[edge])
            extra = self.dimensions.new_tensor([len(keys), len(scope), len(opened), *candidate["estimate"]]).log1p()
            pooled = torch.cat((local_c.sum(0), local_v[role == 0].sum(0), local_v[role == 1].sum(0), extra))
            predictions.append(self.head(pooled))
        return torch.stack(predictions)

    @torch.no_grad()
    def select(self, domains, candidates, *, cost_weight=1.0, contradiction_reward=0.0):
        """Return a candidate index, or None for the zero-score skip action."""
        if not candidates:
            return None
        prediction = self(domains, candidates)
        scores = prediction[:, 0].clamp(0, 20).expm1() - cost_weight * prediction[:, 1].clamp(-20, 20).exp()
        scores += contradiction_reward * prediction[:, 2].sigmoid()
        idx = int(scores.argmax())
        return idx if scores[idx] > 0 else None


def load(path):
    """Reload an own-training checkpoint, without enabling general pickle loading."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != 1:
        raise ValueError("Expected a version-1 predictor checkpoint")
    model = Predictor(checkpoint["network"], **checkpoint["config"])
    model.reward_colors = checkpoint["reward_colors"]
    model.load_state_dict(checkpoint["weights"])
    return model.eval()
