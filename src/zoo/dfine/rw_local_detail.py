"""Query-conditioned P2/P3 local features for ordinary read/write classification.

This is a structural ablation, not an oracle classifier: both training and
inference sample detached *predicted* boxes, never GT crops or GT class IDs.
"""

import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


class RWLocalDetailAdapter(nn.Module):
    """Return a feature residual; leave box regression and LQE computation alone."""

    def __init__(
        self, hidden_dim=256, p2_channels=64, detail_dim=64, grid_size=5,
        num_heads=4, residual_scale=0.1,
    ):
        super().__init__()
        dimensions = (hidden_dim, p2_channels, detail_dim, grid_size, num_heads)
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("RW local detail dimensions must be positive integers")
        if grid_size < 3 or grid_size % 2 == 0:
            raise ValueError("RW local detail grid_size must be odd and >= 3")
        if detail_dim % 8 or detail_dim % num_heads:
            raise ValueError("RW local detail dimension must be divisible by 8 and num_heads")
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("RW local detail residual_scale must be positive and finite")
        self.hidden_dim = hidden_dim
        self.p2_channels = p2_channels
        self.detail_dim = detail_dim
        self.num_heads = num_heads
        self.grid_size = grid_size
        self.residual_scale = float(residual_scale)  # Fixed, not a second zero gate.

        def projection(channels):
            return nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(channels, detail_dim, 1, bias=False)),
                ("norm", nn.GroupNorm(8, detail_dim)), ("act", nn.SiLU()),
            ]))

        # Project whole maps BEFORE ROI sampling: never construct 256-channel
        # per-query feature maps. P2 and P3 keep separate projection weights.
        self.p2_projection = projection(p2_channels)
        self.p3_projection = projection(hidden_dim)
        axis = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size - 0.5
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("relative_grid", torch.stack((xx, yy), -1).flatten(0, 1),
                             persistent=False)
        self.position_projection = nn.Linear(2, detail_dim, bias=False)
        self.scale_embedding = nn.Parameter(torch.zeros(2, detail_dim))
        self.query_projection = nn.Sequential(OrderedDict([
            ("norm", nn.LayerNorm(hidden_dim)), ("linear", nn.Linear(hidden_dim, detail_dim)),
        ]))
        self.token_norm = nn.LayerNorm(detail_dim)
        self.key_projection = nn.Linear(detail_dim, detail_dim)
        self.value_projection = nn.Linear(detail_dim, detail_dim)
        self.fuse = nn.Sequential(OrderedDict([
            ("norm", nn.LayerNorm(2 * detail_dim)),
            ("down", nn.Linear(2 * detail_dim, detail_dim)), ("act", nn.SiLU()),
            ("up", nn.Linear(detail_dim, hidden_dim)),
        ]))
        # Initial outputs reproduce the baseline exactly. Only this projection
        # learns on the first step; the remaining branch starts learning after it.
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def _sample(self, projected, grid):
        batch, queries, tokens, _ = grid.shape
        # grid_sample half-precision coordinate arithmetic can be inaccurate.
        # Keep this small sampling operation in FP32 under CUDA/CPU autocast.
        with torch.autocast(device_type=projected.device.type, enabled=False):
            sampled = F.grid_sample(
                projected.float(), grid.reshape(batch, queries * tokens, 1, 2).float(),
                mode="bilinear", padding_mode="zeros", align_corners=False,
            )
        return sampled.squeeze(-1).transpose(1, 2).reshape(
            batch, queries, tokens, self.detail_dim,
        ).to(projected.dtype)

    def forward(self, p2, p3, boxes, query):
        if query.ndim != 3 or query.shape[-1] != self.hidden_dim:
            raise ValueError("RW local detail query must be [batch, queries, hidden_dim]")
        if boxes.shape != (*query.shape[:2], 4):
            raise ValueError("RW local detail boxes must match query batch/count")
        for name, feature, channels in (("P2", p2, self.p2_channels),
                                        ("P3", p3, self.hidden_dim)):
            if (feature is None or feature.ndim != 4 or feature.shape[0] != query.shape[0]
                    or feature.shape[1] != channels or min(feature.shape[-2:]) <= 0):
                raise ValueError(f"RW local detail requires configured {name} feature channels")
        if query.shape[1] == 0:
            return query.new_zeros(query.shape)

        # Relative locations retain spatial layout. No GT dependency, no score
        # gate, no gradients through box coordinates into the regression head.
        geometry = boxes.detach().float()
        relative = self.relative_grid.to(device=query.device)
        xy = geometry[..., None, :2] + geometry[..., None, 2:] * relative
        grid = 2.0 * xy - 1.0
        p2_tokens = self._sample(self.p2_projection(p2), grid)
        p3_tokens = self._sample(self.p3_projection(p3), grid)
        position = self.position_projection(relative).to(p2_tokens.dtype).unsqueeze(0).unsqueeze(0)
        scale = self.scale_embedding.to(p2_tokens.dtype)
        tokens = torch.cat((p2_tokens + position + scale[0],
                            p3_tokens + position + scale[1]), dim=2)
        tokens = self.token_norm(tokens)
        content = self.query_projection(query)
        batch, count, length, _ = tokens.shape
        head_dim = self.detail_dim // self.num_heads
        q = content.reshape(batch, count, self.num_heads, head_dim)
        keys = self.key_projection(tokens).reshape(
            batch, count, length, self.num_heads, head_dim,
        )
        values = self.value_projection(tokens).reshape_as(keys)
        # Each query attends only inside its own ROI (50 tokens at grid_size=5),
        # not across 300 queries. Compute attention normalization in FP32.
        attention = (q.unsqueeze(2) * keys).sum(-1).float() / math.sqrt(head_dim)
        attention = attention.softmax(dim=2).to(values.dtype)
        local = (attention.unsqueeze(-1) * values).sum(dim=2).flatten(-2)
        return self.residual_scale * self.fuse(torch.cat((content, local), dim=-1))
