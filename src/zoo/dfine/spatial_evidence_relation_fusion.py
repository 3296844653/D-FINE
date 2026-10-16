"""Spatial Evidence Relation Fusion (SERF), a feature-side research candidate.

Learned evidence slots in overlapping P3 windows, P4-conditioned pair relations
with relative spatial layout, then pixel-specific reconstruction of a P3
residual. Slots are NOT supervised anatomical parts. No GT, new loss, box,
query-count or AQS changes. Encoder/Backbone are independent output placements.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

SERF_PLACEMENTS = ("SERF_Encoder", "SERF_Backbone")


class SpatialEvidenceRelationFusion(nn.Module):
    def __init__(self, p3_channels, p4_channels, placement, dim=64, window=8,
                 halo=2, slots=4, heads=4, scale=.1, chunk=128,
                 use_checkpoint=True):
        super().__init__()
        if (placement not in SERF_PLACEMENTS
                or any(type(v) is not int or v < 1 for v in
                       (p3_channels, p4_channels, dim, window, slots, heads, chunk))
                or type(halo) is not int or halo < 0 or halo >= window
                or slots < 2 or dim % 8 or dim % heads or window < 2
                or isinstance(scale, bool) or not math.isfinite(scale) or not 0 < scale < 1
                or type(use_checkpoint) is not bool):
            raise ValueError("Invalid SERF dimensions/settings")
        self.placement = placement
        self.p3_channels, self.p4_channels = p3_channels, p4_channels
        self.dim, self.window, self.halo = dim, window, halo
        self.slots, self.heads, self.chunk = slots, heads, chunk
        self.use_checkpoint = use_checkpoint
        self.enabled = self.geometry_enabled = self.relation_enabled = True
        self.register_buffer("settings", torch.tensor([1, SERF_PLACEMENTS.index(placement),
            p3_channels, p4_channels, dim, window, halo, slots, heads,
            round(scale * 1e9), chunk, int(use_checkpoint)], dtype=torch.int64))
        self.raw_scale = nn.Parameter(torch.tensor(math.atanh(scale)))

        def project(channels):
            return nn.Sequential(nn.Conv2d(channels, dim, 1, bias=False),
                                 nn.GroupNorm(8, dim), nn.SiLU())
        self.p3_projection, self.p4_projection = project(p3_channels), project(p4_channels)
        self.slot_assignment = nn.Conv2d(2 * dim, slots, 1)
        self.relation_qk = nn.Linear(2 * dim, 2 * dim)
        self.relation_value = nn.Linear(dim, dim)
        self.geometry_bias = nn.Sequential(nn.Linear(3, 16), nn.SiLU(), nn.Linear(16, heads))
        self.evidence_update = nn.Sequential(nn.LayerNorm(3 * dim), nn.Linear(3 * dim, dim), nn.SiLU())
        self.pixel_query = nn.Linear(2 * dim, dim)
        self.evidence_key = nn.Linear(2 * dim, dim)
        self.fusion = nn.Sequential(nn.Conv2d(3 * dim, dim, 1, bias=False),
                                    nn.GroupNorm(8, dim), nn.SiLU())
        self.up = nn.Conv2d(dim, p3_channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        side = window + 2 * halo
        axis = (torch.arange(side, dtype=torch.float32) - halo + .5) / window
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("coordinates", torch.stack((xx, yy), -1).reshape(-1, 2), persistent=False)
        self.register_buffer("self_mask", torch.eye(slots, dtype=torch.bool), persistent=False)

    @property
    def scale(self):
        return self.raw_scale.tanh()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        saved = state_dict.get(prefix + "settings")
        if saved is not None and (saved.shape != self.settings.shape
                or not torch.equal(saved.cpu(), self.settings.cpu())):
            error_msgs.append(prefix + "SERF settings differ; use the checkpoint configuration")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _windows(self, tensor):
        b, c, h, w = tensor.shape
        n = (h // self.window) * (w // self.window)
        side = self.window + 2 * self.halo
        columns = F.unfold(tensor, kernel_size=side, stride=self.window, padding=self.halo)
        return columns.transpose(1, 2).reshape(b * n, c, side * side).transpose(1, 2)

    def _restore(self, windows, batch, height, width):
        n = (height // self.window) * (width // self.window)
        side = self.window + 2 * self.halo
        # Only window cores are written. Halo pixels supply context, not copies.
        core = windows.reshape(batch * n, side, side, self.dim)
        core = core[:, self.halo:self.halo + self.window, self.halo:self.halo + self.window]
        columns = core.permute(0, 3, 1, 2).reshape(batch, n, self.dim * self.window ** 2).transpose(1, 2)
        return F.fold(columns, (height, width), kernel_size=self.window, stride=self.window)

    def _interaction(self, p3, p4, assignments, valid):
        """Windows have >=1 real pixel; mask padded pixels before pooling."""
        with torch.autocast(device_type=p3.device.type, enabled=False):
            a = assignments.float().transpose(1, 2).masked_fill(~valid[:, None], -torch.inf).softmax(-1)
            evidence, semantics = a @ p3.float(), a @ p4.float()
            centers = a @ self.coordinates.float()
        joined = torch.cat((evidence, semantics), -1).to(p3.dtype)
        q, k = self.relation_qk(joined).chunk(2, -1)
        head_dim = self.dim // self.heads
        q = q.reshape(-1, self.slots, self.heads, head_dim).transpose(1, 2)
        k = k.reshape(-1, self.slots, self.heads, head_dim).transpose(1, 2)
        delta = centers[:, :, None] - centers[:, None, :]
        geometry = torch.cat((delta, delta.square().sum(-1, keepdim=True)), -1)
        bias = self.geometry_bias(geometry).permute(0, 3, 1, 2)
        with torch.autocast(device_type=p3.device.type, enabled=False):
            relation = q.float() @ k.float().transpose(-1, -2) / math.sqrt(head_dim)
            if self.geometry_enabled:
                relation = relation + bias.float()
            relation = relation.masked_fill(self.self_mask[None, None], -torch.inf).softmax(-1)
        values = self.relation_value(evidence.to(p3.dtype))
        v = values.reshape(-1, self.slots, self.heads, head_dim).transpose(1, 2)
        with torch.autocast(device_type=p3.device.type, enabled=False):
            # Difference from the evidence itself encodes interactions rather
            # than just re-injecting a region's pooled appearance.
            messages = (relation @ v.float() - v.float()).transpose(1, 2).reshape(-1, self.slots, self.dim)
        if not self.relation_enabled:
            messages = torch.zeros_like(messages)
        updated = self.evidence_update(torch.cat((evidence, semantics, messages), -1).to(p3.dtype))
        pixel_query = self.pixel_query(torch.cat((p3, p4), -1))
        keys = self.evidence_key(joined)
        with torch.autocast(device_type=p3.device.type, enabled=False):
            read = (pixel_query.float() @ keys.float().transpose(-1, -2) / math.sqrt(self.dim)).softmax(-1)
            reconstructed = read @ updated.float()
        return reconstructed.to(p3.dtype), a, centers, relation, read

    def _residual(self, p3, p4):
        local = self.p3_projection(p3)
        semantic = self.p4_projection(p4)
        semantic = F.interpolate(semantic, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.slot_assignment(torch.cat((local, semantic), 1))
        b, _, h, w = local.shape
        pad_h, pad_w = (-h) % self.window, (-w) % self.window
        combined = F.pad(torch.cat((local, semantic, logits), 1), (0, pad_w, 0, pad_h))
        windows = self._windows(combined)
        mask = F.pad(torch.ones(b, 1, h, w, device=p3.device, dtype=combined.dtype),
                     (0, pad_w, 0, pad_h))
        valid = self._windows(mask).squeeze(-1).bool()
        reconstructed = []
        for start in range(0, windows.shape[0], self.chunk):
            current = windows[start:start + self.chunk]
            value = self._interaction(current[..., :self.dim], current[..., self.dim:2 * self.dim],
                                      current[..., 2 * self.dim:], valid[start:start + self.chunk])[0]
            reconstructed.append(value)
        evidence = self._restore(torch.cat(reconstructed), b, h + pad_h, w + pad_w)[..., :h, :w]
        # Preserve a local contrast signal and contextual semantics alongside
        # the spatially reconstructed evidence; do not average all three.
        detail = local - F.avg_pool2d(local, 3, stride=1, padding=1, count_include_pad=False)
        residual = self.up(self.fusion(torch.cat((detail, evidence, semantic), 1)))
        return (self.scale * residual).to(p3.dtype)

    def forward(self, p3, p4):
        if (p3.ndim != 4 or p4.ndim != 4 or p3.shape[0] != p4.shape[0]
                or p3.shape[1] != self.p3_channels or p4.shape[1] != self.p4_channels
                or p3.device != p4.device or p3.dtype != p4.dtype
                or min(*p3.shape[-2:], *p4.shape[-2:]) < 1
                or any(c > f for c, f in zip(p4.shape[-2:], p3.shape[-2:]))):
            raise ValueError("SERF expects matched P3/P4 channels, batches and feature hierarchy")
        if not self.enabled:
            return p3
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            residual = checkpoint(self._residual, p3, p4, use_reentrant=False)
        else:
            residual = self._residual(p3, p4)
        return p3 + residual
