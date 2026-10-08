"""Optional medium-scale-oriented detail enhancement for the D-FINE neck.

This is an experimental mechanism, not evidence that a particular network
stage caused the observed errors. It uses feature maps only, never GT boxes,
class labels, confidence thresholds, or validation-set statistics.
"""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def _group_norm(channels):
    # Keep at least two channels per group, including at a 1x1 spatial size.
    groups = next(g for g in (8, 4, 2, 1) if channels % g == 0 and channels // g >= 2)
    return nn.GroupNorm(groups, channels)


class MediumScaleFineGrainedEnhancement(nn.Module):
    """Restore local detail under the guidance of the fused semantic feature.

    ``source`` is a channel-projected pre-fusion backbone feature; ``fused``
    is the FPN feature at the SAME stride. A semantic-conditioned spatial
    softmax mixes depthwise 3x3, 5x5, and 7x7 detail branches. A bounded,
    learnable residual scale starts at 0.1. Unlike a zero-initialized output
    head, all detail/gating parameters can receive gradients on the first step.

    Applied to stride-8/16 P3/P4, it aims to test medium-scale behavior cues;
    feature levels are NOT COCO object-size labels. It also affects other sizes
    and, through the shared encoder, may affect both classification and boxes.
    """

    MAX_RESIDUAL_SCALE = 0.5

    def __init__(self, channels, mid_channels=64, alpha_init=0.1, use_checkpoint=True):
        super().__init__()
        for name, value in (("channels", channels), ("mid_channels", mid_channels)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 4:
                raise ValueError(f"MFFE {name} must be an integer >= 4")
        if not math.isfinite(alpha_init) or not 0 <= alpha_init < self.MAX_RESIDUAL_SCALE:
            raise ValueError("MFFE alpha_init must be finite and in [0, 0.5)")
        self.channels = channels
        self.use_checkpoint = bool(use_checkpoint)
        self.detail_proj = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            _group_norm(mid_channels),
            nn.SiLU(),
        )
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            _group_norm(mid_channels),
            nn.SiLU(),
        )
        self.detail_branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel, padding=kernel // 2,
                          groups=mid_channels, bias=False),
                _group_norm(mid_channels),
                nn.SiLU(),
            )
            for kernel in (3, 5, 7)
        )
        self.branch_gate = nn.Conv2d(2 * mid_channels, 3, 1)
        # Near-uniform initial weights, without blocking the semantic
        # projection's first-step gradients with a zero gate matrix.
        nn.init.normal_(self.branch_gate.weight, std=0.01)
        nn.init.zeros_(self.branch_gate.bias)
        self.restore = nn.Sequential(
            nn.Conv2d(mid_channels, channels, 1, bias=False),
            _group_norm(channels),
        )
        self.raw_alpha = nn.Parameter(torch.tensor(
            math.atanh(float(alpha_init) / self.MAX_RESIDUAL_SCALE), dtype=torch.float32
        ))

    def residual_scale(self):
        """Signed scale, bounded within (-0.5, 0.5) throughout training."""
        return self.MAX_RESIDUAL_SCALE * self.raw_alpha.tanh()

    def forward(self, source, fused):
        if source.ndim != 4 or fused.ndim != 4 or source.shape != fused.shape:
            raise ValueError("MFFE source and fused must be equal-shaped BCHW tensors")
        if source.shape[1] != self.channels:
            raise ValueError("MFFE input channels do not match its configured width")
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            # Recompute ONLY this stateless GroupNorm branch in backward.
            # Do not checkpoint the surrounding baseline BatchNorm/FPN/PAN,
            # which would update their running statistics a second time.
            return checkpoint(self._forward_impl, source, fused, use_reentrant=False)
        return self._forward_impl(source, fused)

    def _forward_impl(self, source, fused):
        detail = self.detail_proj(source)
        semantic = self.semantic_proj(fused)
        # Only the small gate logits are promoted to fp32 for AMP stability.
        weights = self.branch_gate(torch.cat((detail, semantic), dim=1))
        weights = weights.float().softmax(dim=1).to(detail.dtype)
        mixed = sum(
            weights[:, i:i + 1] * branch(detail)
            for i, branch in enumerate(self.detail_branches)
        )
        residual = self.restore(mixed).to(fused.dtype)
        return fused + self.residual_scale().to(fused.dtype) * residual
