"""Three literature-inspired INSTANCE-level read/write evidence experiments.

Not reproductions of PartNet, TASN or iHOI. Only predicted person boxes and
ordinary queries are used; no GT crops, pose, gaze, object or part annotations.
The output is a zero-initialized final-classification residual [0, read, write].
Each experiment is independent and can complement the unchanged L2 AQS.
"""

import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


INSTANCE_EVIDENCE_MODULES = (
    "PartEvidenceSelection", "TrilinearDetailSampling", "LocalInteractionEvidence",
)


class InstanceEvidenceBase(nn.Module):
    def __init__(self, hidden_dim=256, p2_channels=64, dim=64, grid=7,
                 parts=3, scale=.1, chunk_size=32, use_checkpoint=True):
        super().__init__()
        if any(type(x) is not int or x <= 0 for x in
               (hidden_dim, p2_channels, dim, grid, parts, chunk_size)):
            raise ValueError("Evidence dimensions and chunk_size must be positive integers")
        if dim % 8 or grid < 3 or grid % 2 == 0 or not 2 <= parts <= 8:
            raise ValueError("Evidence needs dim divisible by 8, odd grid >=3, and 2..8 parts")
        if not math.isfinite(scale) or scale <= 0 or type(use_checkpoint) is not bool:
            raise ValueError("Evidence scale must be finite/positive; checkpoint must be boolean")
        self.hidden_dim, self.p2_channels, self.dim = hidden_dim, p2_channels, dim
        self.grid, self.parts = grid, parts
        self.scale, self.chunk_size, self.use_checkpoint = float(scale), chunk_size, use_checkpoint
        self.register_buffer("_evidence_settings", torch.tensor([
            INSTANCE_EVIDENCE_MODULES.index(type(self).__name__), hidden_dim, p2_channels,
            dim, grid, parts, round(self.scale * 1_000_000_000),
        ], dtype=torch.int64))  # Integer buffer is immune to EMA averaging / FP16 casts.

        def projection(channels):
            return nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(channels, dim, 1, bias=False)),
                ("norm", nn.GroupNorm(8, dim)), ("act", nn.SiLU()),
            ]))
        self.p2_projection, self.p3_projection = projection(p2_channels), projection(hidden_dim)
        self.token_fusion = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(2 * dim, dim)), ("act", nn.SiLU()), ("norm", nn.LayerNorm(dim)),
        ]))
        self.query_projection = nn.Sequential(OrderedDict([
            ("norm", nn.LayerNorm(hidden_dim)), ("linear", nn.Linear(hidden_dim, dim)),
        ]))
        self.position_projection = nn.Linear(2, dim, bias=False)
        axis = (torch.arange(grid, dtype=torch.float32) + .5) / grid - .5
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("relative_grid", torch.stack((xx, yy), -1).flatten(0, 1),
                             persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        saved = state_dict.get(prefix + "_evidence_settings")
        if saved is not None and (saved.shape != self._evidence_settings.shape or
                not torch.equal(saved.cpu(), self._evidence_settings.cpu())):
            error_msgs.append(prefix + "evidence settings differ from checkpoint; use its original configuration")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def make_output(self, inputs):
        self.output = nn.Sequential(OrderedDict([
            ("norm", nn.LayerNorm(inputs)), ("down", nn.Linear(inputs, self.dim)),
            ("act", nn.SiLU()), ("up", nn.Linear(self.dim, 2)),
        ]))
        nn.init.zeros_(self.output.up.weight)
        nn.init.zeros_(self.output.up.bias)

    @staticmethod
    def clipped_geometry(boxes):
        # Detach ALL coordinate paths. Clipping prevents sampling neighbouring
        # content outside the visible predicted ROI; empty ROIs give zero delta.
        box = boxes.detach().float()
        low = (box[..., :2] - box[..., 2:].clamp_min(0) / 2).clamp(0, 1)
        high = (box[..., :2] + box[..., 2:].clamp_min(0) / 2).clamp(0, 1)
        extent = (high - low).clamp_min(0)
        return torch.cat(((low + high) / 2, extent), -1)

    def sample(self, feature, geometry, relative):
        batch, count = geometry.shape[:2]
        if relative.ndim == 2:
            relative = relative[None, None].expand(batch, count, -1, -1)
        locations = geometry[..., None, :2] + geometry[..., None, 2:] * relative.float()
        with torch.autocast(device_type=feature.device.type, enabled=False):
            sampled = F.grid_sample(feature.float(), (locations * 2 - 1).flatten(1, 2).unsqueeze(2),
                                    align_corners=False, padding_mode="border")
        return sampled.squeeze(-1).transpose(1, 2).reshape(
            batch, count, relative.shape[-2], self.dim).to(feature.dtype)

    def tokens(self, p2, p3, geometry, relative=None):
        relative = self.relative_grid if relative is None else relative
        return self.token_fusion(torch.cat((self.sample(p2, geometry, relative),
                                           self.sample(p3, geometry, relative)), -1))

    def forward(self, query, boxes, p2, p3):
        if query.ndim != 3 or query.shape[-1] != self.hidden_dim:
            raise ValueError("Evidence query must be [batch, queries, hidden_dim]")
        if boxes.shape != (*query.shape[:2], 4):
            raise ValueError("Evidence boxes must match query batch/count")
        for feature, channels in ((p2, self.p2_channels), (p3, self.hidden_dim)):
            if (feature is None or feature.ndim != 4 or feature.shape[:2] != (query.shape[0], channels)
                    or min(feature.shape[-2:]) < 1):
                raise ValueError("Evidence needs configured native P2 and decoder P3 features")
        if query.shape[1] == 0:
            return query.new_zeros((*query.shape[:2], 3))
        geometry = self.clipped_geometry(boxes)
        p2, p3 = self.p2_projection(p2), self.p3_projection(p3)
        pieces = []
        # Chunking bounds per-operation workspace; checkpointing avoids keeping
        # all ROI/channel-relation activations for batch32 x 300 queries.
        for start in range(0, query.shape[1], self.chunk_size):
            q, box = query[:, start:start+self.chunk_size], geometry[:, start:start+self.chunk_size]
            if self.training and self.use_checkpoint and torch.is_grad_enabled():
                pair = checkpoint(self._forward_chunk, q, box, p2, p3, use_reentrant=False)
            else:
                pair = self._forward_chunk(q, box, p2, p3)
            valid = (box[..., 2:] > 0).all(-1, keepdim=True).to(pair.dtype)
            pair = self.scale * pair * valid
            pieces.append(torch.cat((torch.zeros_like(pair[..., :1]), pair), -1))
        return torch.cat(pieces, 1).to(query.dtype)


class PartEvidenceSelection(InstanceEvidenceBase):
    """PartNet-inspired dual region streams, with explicit irrelevant-region mass.

Dense cells and overlapping 3x3 pooled patches are compact region candidates.
Classification produces read/write/other evidence. The independent selector
assigns regions to multiple latent parts OR an irrelevant slot. This is not
PartNet's DPP, BCE supervision or multi-CNN PartNet-Full ensemble.
"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.region_classification = nn.Linear(self.dim, 3)
        self.region_selection = nn.Linear(self.dim, self.parts + 1)
        self.make_output(self.dim + self.parts * (self.dim + 4))

    def region_streams(self, query, p2, p3, geometry):
        dense = self.tokens(p2, p3, geometry)
        b, q, _, d = dense.shape
        coarse = F.avg_pool2d(dense.reshape(b*q, self.grid, self.grid, d).permute(0, 3, 1, 2),
                              3, stride=2).flatten(2).transpose(1, 2).reshape(b, q, -1, d)
        positions = self.relative_grid.reshape(self.grid, self.grid, 2).permute(2, 0, 1)[None]
        coarse_positions = F.avg_pool2d(positions, 3, stride=2).flatten(2).transpose(1, 2)[0]
        coordinates = torch.cat((self.relative_grid, coarse_positions), 0)
        candidates = torch.cat((dense, coarse), 2) + self.position_projection(coordinates)
        conditioned = candidates + query.unsqueeze(2)
        classification = self.region_classification(conditioned).float().softmax(-1)
        assignment = self.region_selection(conditioned).float().softmax(-1)
        relevance = assignment[..., :self.parts].transpose(-1, -2)
        availability = relevance.mean(-1)
        selection = relevance / relevance.sum(-1, keepdim=True).clamp_min(1e-6)
        return candidates, classification, assignment, selection, availability

    def _forward_chunk(self, query, geometry, p2, p3):
        content = self.query_projection(query)
        candidates, classes, _, selection, availability = self.region_streams(content, p2, p3, geometry)
        # FP32 weighted sums prevent softmax/aggregation underflow in FP16.
        with torch.autocast(device_type=query.device.type, enabled=False):
            part_features = torch.matmul(selection, candidates.float()) * availability.unsqueeze(-1)
            evidence = torch.matmul(selection, classes) * availability.unsqueeze(-1)
        fused = torch.cat((content, part_features.flatten(-2), evidence.flatten(-2), availability), -1)
        return self.output(fused)


class TrilinearDetailSampling(InstanceEvidenceBase):
    """TASN-inspired local channel relations and actual inverse-CDF resampling.

Structure/detail attention maps produce TWO nonuniform grids that resample
native P2/P3 maps, not the already sampled coarse ROI. No teacher, KD loss or
whole-image warp. The deterministic query-conditioned detail map replaces
TASN's random channel choice for crowded-instance detection.
"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.channel_query = nn.Linear(self.dim, self.dim)
        self.sample_grid = 5
        self.make_output(4 * self.dim)

    @staticmethod
    def trilinear_attention(tokens):
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            x = tokens.float().transpose(-1, -2)
            relation = torch.matmul(x.softmax(-1), x.transpose(-1, -2)).softmax(-1)
            return torch.matmul(relation, x).softmax(-1)

    @staticmethod
    def inverse_cdf(density, count):
        # Piecewise-linear inverse with detached search indices, but gradients
        # through CDF values and interpolation. A small uniform floor keeps
        # even low-attention areas observable and prevents zero-width bins.
        with torch.autocast(device_type=density.device.type, enabled=False):
            probability = density.float().clamp_min(0)
            probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-8)
            probability = .95 * probability + .05 / probability.shape[-1]
            cdf = probability.cumsum(-1)
            targets = ((torch.arange(count, device=density.device, dtype=torch.float32) + .5) / count)
            targets = targets.expand(*density.shape[:-1], count).contiguous()
            indices = torch.searchsorted(cdf.detach().contiguous(), targets, right=True)
            indices = indices.clamp_max(probability.shape[-1] - 1)
            before = F.pad(cdf, (1, 0))[..., :-1].gather(-1, indices)
            mass = probability.gather(-1, indices).clamp_min(1e-8)
            return ((indices.float() + (targets - before) / mass) / probability.shape[-1] - .5).clamp(-.5, .5)

    def quantile_grid(self, attention):
        side = self.grid
        density = attention.reshape(*attention.shape[:-1], side, side)
        x = self.inverse_cdf(density.amax(-2), self.sample_grid)
        y = self.inverse_cdf(density.amax(-1), self.sample_grid)
        return torch.stack((x.unsqueeze(-2).expand(*x.shape[:-1], self.sample_grid, self.sample_grid),
                            y.unsqueeze(-1).expand(*y.shape[:-1], self.sample_grid, self.sample_grid)), -1).flatten(-3, -2)

    def _forward_chunk(self, query, geometry, p2, p3):
        content = self.query_projection(query)
        attention = self.trilinear_attention(self.tokens(p2, p3, geometry))
        channels = self.channel_query(content).float().softmax(-1)
        structure = attention.mean(-2)
        detail = (attention * channels.unsqueeze(-1)).sum(-2)
        structure_grid, detail_grid = self.quantile_grid(structure), self.quantile_grid(detail)
        structure_tokens = self.tokens(p2, p3, geometry, structure_grid)
        detail_tokens = self.tokens(p2, p3, geometry, detail_grid)
        structure_tokens = structure_tokens + self.position_projection(structure_grid.to(content.dtype))
        detail_tokens = detail_tokens + self.position_projection(detail_grid.to(content.dtype))
        structure_feature, detail_feature = structure_tokens.mean(2), detail_tokens.mean(2)
        return self.output(torch.cat((content, structure_feature, detail_feature,
                                      detail_feature - structure_feature), -1))


class LocalInteractionEvidence(InstanceEvidenceBase):
    """iHOI-inspired directed local relations, WITHOUT literal objects/pose/gaze.

Latent region slots use weak spatial priors, not anatomical labels. Directed
edges include appearance and relative geometry. Read/write independently
select relevant edges; unlike PartInteractionL2, this updates final logits,
not an L2 query, and does not average all pair relations indiscriminately.
"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.role_embedding = nn.Parameter(torch.randn(self.parts, self.dim) * .02)
        if self.parts == 3:
            centers = torch.tensor([[0., -.25], [-.25, .25], [.25, .25]])
        else:
            angle = torch.arange(self.parts) * (2 * math.pi / self.parts)
            centers = .25 * torch.stack((angle.cos(), angle.sin()), -1)
        self.role_centers = nn.Parameter(centers)
        self.region_keys = nn.Linear(self.dim, self.dim)
        self.relation = nn.Sequential(OrderedDict([
            ("norm", nn.LayerNorm(4 * self.dim + 3)),
            ("linear", nn.Linear(4 * self.dim + 3, self.dim)), ("act", nn.SiLU()),
        ]))
        self.class_embedding = nn.Parameter(torch.randn(2, self.dim) * .02)
        self.edge_keys = nn.Linear(self.dim, self.dim)
        edges = torch.tensor([(i, j) for i in range(self.parts) for j in range(self.parts) if i != j])
        self.register_buffer("edges", edges, persistent=False)
        self.make_output(3 * self.dim)

    def interaction_evidence(self, content, tokens):
        coordinates = self.relative_grid.float()
        tokens = tokens + self.position_projection(self.relative_grid)
        roles = content.unsqueeze(2) + self.role_embedding
        keys = self.region_keys(tokens)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            spatial_prior = -4 * (coordinates[None] - self.role_centers.float()[:, None].tanh()).square().sum(-1)
            attention = (torch.matmul(roles.float(), keys.float().transpose(-1, -2)) / math.sqrt(self.dim)
                         + spatial_prior).softmax(-1)
            features = torch.matmul(attention, tokens.float())
            positions = torch.matmul(attention, coordinates)
        a, b = features[..., self.edges[:, 0], :], features[..., self.edges[:, 1], :]
        displacement = positions[..., self.edges[:, 1], :] - positions[..., self.edges[:, 0], :]
        geometry = torch.cat((displacement, displacement.square().sum(-1, keepdim=True)), -1)
        relation = self.relation(torch.cat((a, b, a-b, a*b, geometry), -1))
        keys = self.edge_keys(relation)
        receivers = content.unsqueeze(2) + self.class_embedding
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            weights = (torch.matmul(receivers.float(), keys.float().transpose(-1, -2)) / math.sqrt(self.dim)).softmax(-1)
            evidence = torch.matmul(weights, relation.float())
        return evidence, attention, weights, positions

    def _forward_chunk(self, query, geometry, p2, p3):
        content = self.query_projection(query)
        evidence, _, _, _ = self.interaction_evidence(content, self.tokens(p2, p3, geometry))
        return self.output(torch.cat((content, evidence.flatten(-2)), -1))


def build_instance_evidence(name, **kwargs):
    if name not in INSTANCE_EVIDENCE_MODULES:
        raise ValueError(f"Unknown instance evidence module: {name}")
    return globals()[name](**kwargs)
