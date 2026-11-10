"""Supervised anchor choice and whole-cluster masks, without proposal generation.

The anchor is a variable actually pruned by a teacher application plus GAC. It
need not occur in the cluster or its outputs: propagation can carry information
elsewhere. Count heads decode sparse masks without a fixed cluster cardinality.
"""
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .candidate_evaluation_tnreason import SummaryClusterCandidate
from .encoder import Encoder, _mlp


def _balanced_bce(logits, target):
    loss = F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype), reduction="none")
    positive, negative = target.bool(), ~target.bool()
    groups = positive.any(1).to(logits.dtype) + negative.any(1).to(logits.dtype)
    return ((loss * positive).sum(1) / positive.sum(1).clamp_min(1)
            + (loss * negative).sum(1) / negative.sum(1).clamp_min(1)) / groups.clamp_min(1)


def _membership_loss(logits, target):
    # Top-k decoding needs correct relative scores. Averaging thousands of
    # negative BCE terms alone can hide a few very confident wrong members.
    count = target.sum(1)
    ranking = -(logits.log_softmax(-1) * target).sum(1) / count.clamp_min(1)
    ranking = torch.where(count > 0, ranking - count.clamp_min(1).log(), 0.)
    return (_balanced_bce(logits, target) + .5 * ranking).mean()


class MaskPolicy(Encoder):
    kind = "construction"  # Reuse sequential temporary-summary MAC inference.

    def __init__(self, network, width=32, rounds=3, *, anchor_rounds=3,
                 max_cores=64, max_open_colors=6, max_work=1_000_000,
                 max_size=100_000, cluster_attempts=64):
        super().__init__(network, width, rounds, normalize=True)
        if any(type(value) is not int or value < 1 for value in
               (anchor_rounds, max_cores, max_open_colors, max_work, max_size, cluster_attempts)):
            raise ValueError("Expected positive mask-model budgets and anchor rounds")
        self.config.update(anchor_rounds=anchor_rounds, max_cores=max_cores,
            max_open_colors=max_open_colors, max_work=max_work, max_size=max_size,
            cluster_attempts=cluster_attempts)
        self.keys = [core["key"] for core in network["cores"]]
        self.anchor_head = _mlp(3 * width, width, 1)
        self.anchor_marker = nn.Parameter(torch.randn(width) * .1)
        self.core_head = _mlp(4 * width, width, 1)
        self.open_head = _mlp(4 * width, width, 1)
        self.count_head = _mlp(3 * width, width, max_cores + max_open_colors + 2)

    def _round_batch(self, variables, cores):
        vi, ci = self.variable_idx, self.core_idx
        axes = self.axes[None].expand(variables.shape[0], -1, -1)
        def update(nodes, messages, idx, layer):
            incoming = torch.zeros_like(nodes).index_add(1, idx, messages)
            degree = torch.bincount(idx, minlength=nodes.shape[1]).clamp_min(1)
            incoming = incoming / degree[None, :, None]
            result = nodes + layer(torch.cat((nodes, incoming), dim=-1))
            return F.layer_norm(result, (result.shape[-1],))
        messages = self.to_core(torch.cat((variables[:, vi], cores[:, ci], axes), dim=-1))
        cores = update(cores, messages, ci, self.update_core)
        messages = self.to_variable(torch.cat((cores[:, ci], variables[:, vi], axes), dim=-1))
        return update(variables, messages, vi, self.update_variable), cores

    def encode_allowed(self, allowed):
        """True graph batching; allowed has shape (batch, variables, max_dim)."""
        if allowed.ndim != 3 or tuple(allowed.shape[1:]) != (len(self.dimensions), self.max_dim):
            raise ValueError("Expected batched complete domain masks")
        mask = allowed.to(self.dimensions.dtype)
        if torch.any(mask.bool() & ~self.valid) or torch.any(mask.sum(-1) == 0):
            raise ValueError("Expected nonempty domains within variable dimensions")
        scalar = torch.stack((self.dimensions / self.max_dim,
            torch.zeros_like(self.dimensions), (self.dimensions == 2).float(), self.degree.log1p()), dim=-1)
        scalar = scalar[None].expand(len(mask), -1, -1).clone()
        scalar[:, :, 1] = mask.sum(-1) / self.dimensions
        features = torch.cat((mask, self.valid[None].expand(len(mask), -1, -1), scalar), dim=-1)
        variables = self.variable_encoder(features)
        cores = self.core_encoder(self.core_features)[None].expand(len(mask), -1, -1)
        for _ in range(self.config["rounds"]):
            variables, cores = self._round_batch(variables, cores)
        return variables, cores

    def forward(self, allowed, anchors=None):
        variables, cores = self.encode_allowed(allowed)
        batch, count = variables.shape[:2]
        global_state = torch.cat((variables.mean(1), cores.mean(1)), dim=-1)
        anchor_logits = self.anchor_head(torch.cat((variables,
            global_state[:, None].expand(-1, count, -1)), dim=-1)).squeeze(-1)
        eligible = allowed.sum(-1) > 1
        # Fully solved states never enter the runtime; keep forward well-defined.
        eligible = eligible | ~eligible.any(1, keepdim=True)
        anchor_logits = anchor_logits.masked_fill(~eligible, -torch.inf)
        if anchors is None:
            anchors = anchor_logits.argmax(1)
        anchors = torch.as_tensor(anchors, dtype=torch.long, device=variables.device)
        if anchors.shape != (batch,) or torch.any(anchors < 0) or torch.any(anchors >= count):
            raise ValueError("Expected one valid anchor index per state")
        marker = F.one_hot(anchors, count).to(variables.dtype)
        variables = variables + marker[:, :, None] * self.anchor_marker
        for _ in range(self.config["anchor_rounds"]):
            variables, cores = self._round_batch(variables, cores)
        context = torch.cat((variables.mean(1), cores.mean(1),
            variables[torch.arange(batch, device=variables.device), anchors]), dim=-1)
        core_logits = self.core_head(torch.cat((cores,
            context[:, None].expand(-1, len(self.keys), -1)), dim=-1)).squeeze(-1)
        open_logits = self.open_head(torch.cat((variables,
            context[:, None].expand(-1, count, -1)), dim=-1)).squeeze(-1)
        counts = self.count_head(context)
        split = self.config["max_cores"] + 1
        core_counts, open_counts = counts[:, :split], counts[:, split:]
        core_counts = core_counts.masked_fill(
            torch.arange(split, device=counts.device) > len(self.keys), -torch.inf)
        open_counts = open_counts.masked_fill(
            torch.arange(open_counts.shape[1], device=counts.device) > count, -torch.inf)
        return dict(anchor_logits=anchor_logits, core_logits=core_logits, open_logits=open_logits,
            core_count_logits=core_counts, open_count_logits=open_counts, anchors=anchors)

    def loss(self, allowed, core_target, output_target, pruned_target, anchors=None):
        """Positive anchor imitation + coherent full-mask labels, not a union of clusters."""
        core_target, output_target, pruned_target = (target.bool() for target in
            (core_target, output_target, pruned_target))
        positive = pruned_target & (allowed.sum(-1) > 1)
        if (positive.shape != allowed.shape[:2] or torch.any(~positive.any(1))
                or core_target.shape != (len(allowed), len(self.keys))
                or output_target.shape != allowed.shape[:2]):
            raise ValueError("Expected complete teacher masks and at least one actually pruned variable")
        if anchors is None:
            anchors = torch.multinomial(positive.float(), 1).squeeze(1)
        anchors = torch.as_tensor(anchors, dtype=torch.long, device=allowed.device)
        if (anchors.shape != (len(allowed),) or torch.any(anchors < 0)
                or torch.any(anchors >= allowed.shape[1])
                or not torch.all(positive.gather(1, anchors[:, None]))):
            raise ValueError("Teacher anchors must have been pruned by this application")
        prediction = self(allowed, anchors)
        logp = prediction["anchor_logits"].log_softmax(-1)
        anchor_loss = -torch.logsumexp(logp.masked_fill(~positive, -torch.inf), dim=1).mean()
        core_loss = _membership_loss(prediction["core_logits"], core_target)
        output_loss = _membership_loss(prediction["open_logits"], output_target)
        nc, no = core_target.sum(1), output_target.sum(1)
        if torch.any(nc == 0) or torch.any(nc > self.config["max_cores"]) or torch.any(no > self.config["max_open_colors"]):
            raise ValueError("Teacher masks exceed model construction limits")
        count_loss = F.cross_entropy(prediction["core_count_logits"], nc) + F.cross_entropy(
            prediction["open_count_logits"], no)
        loss = anchor_loss + core_loss + output_loss + .25 * count_loss
        with torch.no_grad():
            chosen = prediction["anchor_logits"].argmax(1)
            core_rank = prediction["core_logits"].argsort(dim=1, descending=True, stable=True).argsort(dim=1)
            output_rank = prediction["open_logits"].argsort(dim=1, descending=True, stable=True).argsort(dim=1)
            core_mask = core_rank < prediction["core_count_logits"].argmax(1)[:, None]
            output_mask = output_rank < prediction["open_count_logits"].argmax(1)[:, None]
            metrics = dict(anchor_loss=anchor_loss.detach(), core_loss=core_loss.detach(),
                output_loss=output_loss.detach(), count_loss=count_loss.detach(),
                anchor_hit=positive.gather(1, chosen[:, None]).float().mean(),
                core_count_accuracy=(prediction["core_count_logits"].argmax(1) == nc).float().mean(),
                output_count_accuracy=(prediction["open_count_logits"].argmax(1) == no).float().mean(),
                teacher_core_recall=((core_rank < nc[:, None]) & core_target).sum(1).div(nc).mean(),
                teacher_output_recall=((output_rank < no[:, None]) & output_target).sum(1).div(no.clamp_min(1)).mean(),
                teacher_exact_masks=((core_mask == core_target).all(1)
                    & (output_mask == output_target).all(1)).float().mean())
        return loss, metrics

    @torch.no_grad()
    def propose(self, domains):
        if isinstance(domains, Mapping):
            domains = [domains[color] for color in self.network["colors"]]
        # Reuse the shared encoder's schema/value checks for runtime inputs.
        allowed = self.domain_features(domains)[:, :self.max_dim].bool()[None]
        if not torch.any(allowed.sum(-1) > 1):
            return None
        prediction = self(allowed)
        nc, no = (int(prediction[name].argmax(1)) for name in ("core_count_logits", "open_count_logits"))
        if not nc:
            return None
        cores = prediction["core_logits"][0].argsort(descending=True, stable=True)[:nc].tolist()
        scope = sorted({variable for core in cores for variable in self.scopes[core]})
        if no > len(scope):
            return None
        order = prediction["open_logits"][0, scope].argsort(descending=True, stable=True)[:no].tolist()
        opened = sorted(scope[index] for index in order)
        return SummaryClusterCandidate("mask", tuple(self.keys[i] for i in sorted(cores)),
            tuple(self.network["colors"][i] for i in opened))
