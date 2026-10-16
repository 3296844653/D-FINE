"""Read/write hypothesis-conditioned evidence -> L2 query residual.

Research hypothesis, not a proven innovation or PartNet/TASN reproduction.
Predicted ROIs only; no GT/part/object labels, extra losses or score boosts.
"""
import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class DiscriminativeEvidenceL2(nn.Module):
    def __init__(self, hidden_dim=256, p2_channels=64, dim=64, grid=7,
                 scale=.1, chunk_size=32, use_checkpoint=True):
        super().__init__()
        if any(type(x) is not int or x <= 0 for x in
               (hidden_dim, p2_channels, dim, grid, chunk_size)):
            raise ValueError("Evidence dimensions must be positive integers")
        if dim % 8 or not 3 <= grid <= 9 or grid % 2 == 0:
            raise ValueError("Evidence dim must be divisible by 8; odd grid in [3,9]")
        if (isinstance(scale, bool) or not math.isfinite(scale) or not 0 < scale <= .5
                or type(use_checkpoint) is not bool):
            raise ValueError("Evidence scale in (0,.5]; checkpoint must be boolean")
        self.hidden_dim, self.p2_channels, self.dim = hidden_dim, p2_channels, dim
        self.grid, self.scale = grid, float(scale)
        self.chunk_size, self.use_checkpoint = chunk_size, use_checkpoint
        # Runtime ablation AFTER strict checkpoint loading. Not a saved setting.
        self.enabled = True
        self.register_buffer("settings", torch.tensor([
            1, hidden_dim, p2_channels, dim, grid, round(scale * 1e9), 2,
        ], dtype=torch.int64))
        axis = (torch.arange(grid, dtype=torch.float32) + .5) / grid - .5
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("relative_grid", torch.stack((xx, yy), -1).flatten(0, 1),
                             persistent=False)

        def project(channels):
            return nn.Sequential(OrderedDict(conv=nn.Conv2d(channels, dim, 1, bias=False),
                norm=nn.GroupNorm(8, dim), act=nn.SiLU()))
        self.p2_projection, self.p3_projection = project(p2_channels), project(hidden_dim)
        self.query_projection = nn.Sequential(OrderedDict(norm=nn.LayerNorm(hidden_dim),
            linear=nn.Linear(hidden_dim, dim)))
        # Learn the fine/semantic mixture per query/channel, using DETACHED
        # predicted box size. No COCO medium/large GT labels are available here.
        self.scale_gate = nn.Linear(dim + 2, 2 * dim)
        self.position_projection = nn.Linear(2, dim, bias=False)
        self.token_projection = nn.Sequential(OrderedDict(norm=nn.LayerNorm(2*dim),
            linear=nn.Linear(2*dim, dim), act=nn.SiLU()))
        self.keys = nn.Linear(dim, dim)
        self.hypotheses = nn.Parameter(torch.randn(2, dim) * .02)
        self.hypothesis_projection = nn.Linear(dim, dim)
        self.null_key = nn.Parameter(torch.zeros(dim))
        self.null_bias = nn.Parameter(torch.zeros(2))
        # Different from earlier final-logit heads and pooled BoxLocal adapters:
        # independent read/write evidence, explicit signed contrast, null mass,
        # and a full-width query update AFTER AQS but BEFORE the final decoder.
        self.output = nn.Sequential(OrderedDict(norm=nn.LayerNorm(5*dim + 2),
            down=nn.Linear(5*dim + 2, dim), act=nn.SiLU(), up=nn.Linear(dim, hidden_dim)))
        nn.init.zeros_(self.output.up.weight)
        nn.init.zeros_(self.output.up.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        saved = state_dict.get(prefix + "settings")
        if saved is not None and not torch.equal(saved.cpu(), self.settings.cpu()):
            error_msgs.append(prefix + "DiscriminativeEvidenceL2 settings differ from checkpoint")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    @staticmethod
    def geometry(boxes):
        box = boxes.detach().float()
        low = (box[..., :2] - box[..., 2:].clamp_min(0)/2).clamp(0, 1)
        high = (box[..., :2] + box[..., 2:].clamp_min(0)/2).clamp(0, 1)
        extent = (high-low).clamp_min(0)
        return torch.cat(((low+high)/2, extent), -1)

    def sample(self, feature, geometry):
        b, q = geometry.shape[:2]
        locations = geometry[..., None, :2] + geometry[..., None, 2:] * self.relative_grid.float()
        with torch.autocast(device_type=feature.device.type, enabled=False):
            value = F.grid_sample(feature.float(), (2*locations-1).flatten(1, 2).unsqueeze(2),
                                  align_corners=False, padding_mode="border")
        return value.squeeze(-1).transpose(1, 2).reshape(b, q, self.grid**2, self.dim).to(feature.dtype)

    def evidence(self, query, geometry, p2, p3, class_vectors):
        """Return evidence and auditable weights; no anatomical-part claims."""
        content = self.query_projection(query)
        fine, semantic = self.sample(p2, geometry), self.sample(p3, geometry)
        b, q, n, d = fine.shape
        with torch.autocast(device_type=query.device.type, enabled=False):
            patch = fine.float().reshape(b*q, self.grid, self.grid, d).permute(0, 3, 1, 2)
            contrast = patch - F.avg_pool2d(patch, 3, 1, 1, count_include_pad=False)
            contrast = contrast.flatten(2).transpose(1, 2).reshape(b, q, n, d)
        gate = self.scale_gate(torch.cat((content, geometry[..., 2:].to(content.dtype)), -1))
        gate = gate.float().reshape(b, q, 2, d).softmax(-2)
        mixed = gate[:, :, 0, None]*fine.float() + gate[:, :, 1, None]*semantic.float()
        tokens = self.token_projection(torch.cat((mixed, contrast), -1).to(content.dtype))
        tokens = tokens + self.position_projection(self.relative_grid.to(content.dtype))
        # Anchor the two slots to the ORIGINAL L2 read/write classifier axes,
        # not arbitrary unnamed part slots. Detached weights avoid a new
        # backward route into the classifier; the original losses still train
        # that classifier. Attention remains weakly supervised, not certified
        # anatomical evidence or a guaranteed interpretable class prototype.
        anchors = self.query_projection(class_vectors.detach())
        hypotheses = self.hypothesis_projection(content.unsqueeze(2) + anchors + self.hypotheses)
        keys = self.keys(tokens)
        with torch.autocast(device_type=query.device.type, enabled=False):
            logits = hypotheses.float() @ keys.float().transpose(-1, -2) / math.sqrt(d)
            null = (hypotheses.float()*self.null_key.float()).sum(-1)/math.sqrt(d) + self.null_bias.float()
            weights = torch.cat((logits, null.unsqueeze(-1)), -1).softmax(-1)
            # Null value is EXACTLY zero. Keep the mass (do not renormalize
            # real cells), so attention can explicitly decline local evidence.
            pair = weights[..., :-1] @ tokens.float()
        return content, pair, weights, gate

    def _forward_chunk(self, query, geometry, scores, p2, p3, class_vectors):
        content, pair, weights, _ = self.evidence(query, geometry, p2, p3, class_vectors)
        read, write = pair.unbind(-2)
        fused = torch.cat((content.float(), read, write, write-read, read*write,
                           1-weights[..., -1]), -1)
        delta = self.output(fused.to(content.dtype))
        # Soft ambiguity emphasis, NOT a hard confidence/class selector. Every
        # query gets >=.5 of the residual, including low-score writing queries.
        p = scores.detach().float()[..., 1:3].softmax(-1)[..., :1]
        ambiguity = .5 + 2*p*(1-p)
        availability = (1-weights[..., -1]).mean(-1, keepdim=True)
        valid = (geometry[..., 2:] > 0).all(-1, keepdim=True)
        # If BOTH hypotheses defer to null, even the output bias/content path
        # is suppressed. This is a structural no-evidence safeguard, not a
        # guarantee that learned null mass identifies real false positives.
        return delta * (self.scale*ambiguity*availability*valid).to(delta.dtype)

    def forward(self, query, boxes, scores, p2, p3, class_vectors):
        if query.ndim != 3 or query.shape[-1] != self.hidden_dim:
            raise ValueError("Evidence query must be [batch, queries, hidden_dim]")
        if boxes.shape != (*query.shape[:2], 4) or scores.shape != (*query.shape[:2], 3):
            raise ValueError("Evidence boxes/scores must match query batch/count")
        if class_vectors.shape != (2, self.hidden_dim):
            raise ValueError("Evidence requires L2 read/write classifier vectors")
        if not self.enabled or query.shape[1] == 0:
            return query
        for feature, channels in ((p2, self.p2_channels), (p3, self.hidden_dim)):
            if (feature is None or feature.ndim != 4 or feature.shape[:2] != (query.shape[0], channels)
                    or min(feature.shape[-2:]) < 1):
                raise ValueError("Evidence requires native P2 and projected P3")
        geometry = self.geometry(boxes)
        p2, p3 = self.p2_projection(p2), self.p3_projection(p3)
        pieces = []
        for start in range(0, query.shape[1], self.chunk_size):
            end = start+self.chunk_size
            args = (query[:, start:end], geometry[:, start:end], scores[:, start:end], p2, p3, class_vectors)
            if self.training and self.use_checkpoint and torch.is_grad_enabled():
                delta = checkpoint(self._forward_chunk, *args, use_reentrant=False)
            else:
                delta = self._forward_chunk(*args)
            pieces.append(query[:, start:end]+delta.to(query.dtype))
        return torch.cat(pieces, 1)
