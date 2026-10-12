"""Scale-Aligned Detail Fusion (SADF), a research candidate, not proven novelty.

The pre-fusion P3 supplies signed local contrast. Final encoder P3 anchors
bounded P4/P5 alignment, disagreement-aware scale competition, and coupled
detail/context gates. Only final P3 is changed, after FPN/PAN. No GT, image
class labels, queries, or detection confidence are used by this module.

Related ideas: FaPN (ICCV 2021, feature alignment), ASFF (arXiv:1911.09516,
spatial scale fusion), and FocalNet (NeurIPS 2022, gated contextualization).
SADF is not a reproduction of these methods. Its proposed contribution must
be established by nearest-method comparisons, controlled ablations and repeats.
"""

import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _projection(channels, dim):
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(channels, dim, 1, bias=False)),
        ("norm", nn.GroupNorm(8, dim)),
        ("act", nn.SiLU()),
    ]))


def _zero(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class BoundedScaleAlignment(nn.Module):
    """A displacement field in P3 PIXELS, not coarse pixels/normalized units."""

    def __init__(self, dim, offset_limit):
        super().__init__()
        self.offset_limit = float(offset_limit)
        self.offset = nn.Sequential(OrderedDict([
            ("compress", nn.Conv2d(3 * dim, dim, 1, bias=False)),
            ("norm", nn.GroupNorm(8, dim)),
            ("act", nn.SiLU()),
            ("local", nn.Conv2d(dim, dim, 3, padding=1, groups=dim)),
            ("predict", nn.Conv2d(dim, 2, 1)),
        ]))
        _zero(self.offset.predict)

    @staticmethod
    def sample(coarse, displacement):
        """Zero displacement equals align_corners=False bilinear upsampling."""
        batch, _, height, width = displacement.shape
        # FP32 coordinate/sampling math prevents half-precision pixel rounding.
        with torch.autocast(device_type=coarse.device.type, enabled=False):
            yy = 2 * (torch.arange(height, device=coarse.device, dtype=torch.float32) + .5) / height - 1
            xx = 2 * (torch.arange(width, device=coarse.device, dtype=torch.float32) + .5) / width - 1
            gy, gx = torch.meshgrid(yy, xx, indexing="ij")
            grid = torch.stack((gx, gy), -1).unsqueeze(0).expand(batch, -1, -1, -1)
            scale = displacement.new_tensor([2 / width, 2 / height], dtype=torch.float32)
            grid = grid + displacement.float().permute(0, 2, 3, 1) * scale
            result = F.grid_sample(coarse.float(), grid, mode="bilinear",
                                   padding_mode="border", align_corners=False)
        return result.to(coarse.dtype)

    def forward(self, fine, coarse, detail, enabled=True):
        reference = F.interpolate(coarse, size=fine.shape[-2:], mode="bilinear", align_corners=False)
        # Compute the predictor in ablation mode too: no unused trainable
        # parameters/DDP branches; zero offsets remove only the alignment effect.
        displacement = self.offset(torch.cat((fine, reference, detail), 1))
        displacement = self.offset_limit * displacement.float().tanh()
        if not enabled:
            displacement = displacement * 0
        return self.sample(coarse, displacement), displacement


class ScaleAlignedDetailFusion(nn.Module):
    """A full contrast -> alignment -> scale selection -> joint fusion mechanism.

    The original encoder path and its initialization are preserved. A nonzero
    bounded channel scale multiplies a ZERO-initialized output projection, so
    the initial detector is exactly the baseline. Early gradients first train
    this projection; internal components start receiving nonzero gradients as
    soon as the projection moves. This is intentional, not a dead branch.

    Feature strides are not COCO object-size labels. This module may affect
    all classes/sizes, query initialization, classification AND later boxes.
    """

    MAX_RESIDUAL_SCALE = .5

    def __init__(self, channels=256, dim=64, offset_limit=2., residual_init=.1,
                 use_checkpoint=True, enable_alignment=True, enable_detail=True,
                 enable_scale_selection=True):
        super().__init__()
        for name, value in (("channels", channels), ("dim", dim)):
            if type(value) is not int or value < 16 or value % 8:
                raise ValueError(f"SADF {name} must be an integer >=16 divisible by 8")
        if not math.isfinite(offset_limit) or not 0 < offset_limit <= 4:
            raise ValueError("SADF offset_limit must be in (0,4] P3 pixels")
        if not math.isfinite(residual_init) or not 0 < residual_init < self.MAX_RESIDUAL_SCALE:
            raise ValueError("SADF residual_init must be in (0,0.5), avoiding a double-zero initialization")
        for name, value in (("use_checkpoint", use_checkpoint), ("enable_alignment", enable_alignment),
                            ("enable_detail", enable_detail), ("enable_scale_selection", enable_scale_selection)):
            if type(value) is not bool:
                raise ValueError(f"SADF {name} must be boolean")
        self.channels, self.dim = channels, dim
        self.use_checkpoint = use_checkpoint
        self.enable_alignment, self.enable_detail = enable_alignment, enable_detail
        self.enable_scale_selection = enable_scale_selection
        self.source_projection = _projection(channels, dim)
        self.fine_projection = _projection(channels, dim)
        self.coarse_projections = nn.ModuleList([_projection(channels, dim) for _ in range(2)])
        self.detail_filters = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel, padding=kernel // 2, groups=dim, bias=False)
            for kernel in (3, 5)
        ])
        for layer in self.detail_filters:
            # Start as signed local contrast rather than random edge filters.
            nn.init.zeros_(layer.weight)
            with torch.no_grad():
                layer.weight[:, 0, layer.kernel_size[0] // 2, layer.kernel_size[1] // 2] = 1
        self.detail_selector = nn.Conv2d(3 * dim, 2, 1)
        nn.init.normal_(self.detail_selector.weight, std=.01)
        nn.init.zeros_(self.detail_selector.bias)
        self.alignments = nn.ModuleList([BoundedScaleAlignment(dim, offset_limit) for _ in range(2)])
        self.scale_selector = nn.Sequential(OrderedDict([
            ("compress", nn.Conv2d(4 * dim, dim, 1, bias=False)),
            ("norm", nn.GroupNorm(8, dim)),
            ("act", nn.SiLU()),
            ("score", nn.Conv2d(dim, 3, 1)),
        ]))
        nn.init.normal_(self.scale_selector.score.weight, std=.01)
        nn.init.zeros_(self.scale_selector.score.bias)
        self.joint_gates = nn.Sequential(OrderedDict([
            ("compress", nn.Conv2d(3 * dim, dim, 1, bias=False)),
            ("norm", nn.GroupNorm(8, dim)),
            ("act", nn.SiLU()),
            ("predict", nn.Conv2d(dim, 2 * dim, 1)),
        ]))
        self.fusion = nn.Sequential(OrderedDict([
            ("compress", nn.Conv2d(3 * dim, dim, 1, bias=False)),
            ("norm", nn.GroupNorm(8, dim)),
            ("act", nn.SiLU()),
            ("local", nn.Conv2d(dim, dim, 3, padding=1, groups=dim)),
            ("local_act", nn.SiLU()),
        ]))
        self.restore = nn.Conv2d(dim, channels, 1)
        _zero(self.restore)
        self.raw_alpha = nn.Parameter(torch.full((channels,), math.atanh(residual_init / self.MAX_RESIDUAL_SCALE)))
        # Fixed settings are recoverable from model AND EMA checkpoints.
        self.register_buffer("settings", torch.tensor([
            1, channels, dim, offset_limit, self.MAX_RESIDUAL_SCALE,
            int(enable_alignment), int(enable_detail), int(enable_scale_selection),
            int(use_checkpoint),
        ], dtype=torch.float32))

    def residual_scale(self):
        return self.MAX_RESIDUAL_SCALE * self.raw_alpha.tanh()

    def forward(self, source_p3, p3, p4, p5, return_aux=False):
        if source_p3.ndim != 4 or source_p3.shape != p3.shape or p3.shape[1] != self.channels:
            raise ValueError("SADF pre-fusion source and final P3 must be equal-shaped BCHW tensors")
        for feature in (p4, p5):
            if feature.ndim != 4 or feature.shape[:2] != p3.shape[:2]:
                raise ValueError("SADF P3/P4/P5 must have equal batch/channel dimensions")
            if feature.device != p3.device or feature.dtype != p3.dtype:
                raise ValueError("SADF feature devices/dtypes must agree")
        if source_p3.device != p3.device or source_p3.dtype != p3.dtype:
            raise ValueError("SADF source and P3 devices/dtypes must agree")
        if (p4.shape[-2] > p3.shape[-2] or p4.shape[-1] > p3.shape[-1]
                or p5.shape[-2] > p4.shape[-2] or p5.shape[-1] > p4.shape[-1]):
            raise ValueError("SADF expects spatially decreasing P3/P4/P5")
        if self.use_checkpoint and self.training and torch.is_grad_enabled() and not return_aux:
            # Only stateless GN/new branch is recomputed, never baseline BN/PAN.
            return checkpoint(self._forward_impl, source_p3, p3, p4, p5, use_reentrant=False)
        return self._forward_impl(source_p3, p3, p4, p5, return_aux=return_aux)

    def _forward_impl(self, source_p3, p3, p4, p5, return_aux=False):
        source, fine = self.source_projection(source_p3), self.fine_projection(p3)
        details = [layer(source - F.avg_pool2d(source, kernel, stride=1,
                   padding=kernel // 2, count_include_pad=False))
                   for layer, kernel in zip(self.detail_filters, (3, 5))]
        detail_weights = self.detail_selector(torch.cat((fine, details[0].abs(), details[1].abs()), 1))
        detail_weights = detail_weights.float().softmax(1).to(fine.dtype)
        detail = sum(detail_weights[:, i:i+1] * value for i, value in enumerate(details))
        if not self.enable_detail:
            detail = detail * 0
        aligned, offsets = [], []
        for feature, projection, alignment in zip((p4, p5), self.coarse_projections, self.alignments):
            value, displacement = alignment(fine, projection(feature), detail, self.enable_alignment)
            aligned.append(value); offsets.append(displacement)
        # Explicit discrepancies, not bare upsample+addition. Self-scale P3 is
        # one competitor, so rejecting coarse context yields zero correction.
        selection = torch.cat((fine, (fine-aligned[0]).abs(), (fine-aligned[1]).abs(), detail.abs()), 1)
        scale_logits = self.scale_selector(selection)
        if not self.enable_scale_selection:
            scale_logits = scale_logits * 0
        scale_weights = scale_logits.float().softmax(1).to(fine.dtype)
        context = scale_weights[:, :1] * fine + sum(
            scale_weights[:, i+1:i+2] * value for i, value in enumerate(aligned))
        context_delta = context - fine
        gates = self.joint_gates(torch.cat((fine, context_delta.abs(), detail.abs()), 1)).sigmoid()
        detail_gate, context_gate = gates.chunk(2, 1)
        supported_detail, supported_context = detail_gate * detail, context_gate * context_delta
        interaction = supported_detail * supported_context
        residual = self.restore(self.fusion(torch.cat((supported_detail, supported_context, interaction), 1)))
        result = p3 + self.residual_scale().view(1, -1, 1, 1).to(p3.dtype) * residual.to(p3.dtype)
        if return_aux:
            return result, {"detail_weights": detail_weights, "scale_weights": scale_weights,
                            "detail": detail, "context_delta": context_delta,
                            "detail_gate": detail_gate, "context_gate": context_gate,
                            "offset_p4": offsets[0], "offset_p5": offsets[1]}
        return result
