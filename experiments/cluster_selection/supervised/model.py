"""Supervised cluster predictor with a shared state encoder and local scorer."""
import torch
from torch import nn
from ..encoder import Encoder


class Predictor(Encoder):
    """Predict log1p(gain), log(seconds), and contradiction logit, in that order.

    The ordered raw network fixes the schema. Domains may be ordered lists or a
    mapping keyed by color; candidates use core/color indices and shape estimates.
    Weights are shared across rounds. Tensor axes are never silently reordered.
    """

    def __init__(self, network, width=64, rounds=3):
        super().__init__(network, width, rounds)
        self.roles = nn.Embedding(2, width)
        self.head = nn.Sequential(nn.Linear(3 * width + 6, width), nn.ReLU(),
                                  nn.Linear(width, 32), nn.ReLU(), nn.Linear(32, 3))

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
