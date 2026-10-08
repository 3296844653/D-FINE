"""Opt-in, detached calibration of final read/write query-class scores.

This branch does not change boxes, query selection, or the original training
outputs. Overlapping ordinary queries provide context, not additional targets.
"""

import math

import torch
from torch import nn


class RWQueryCalibration(nn.Module):
    def __init__(self, hidden_dim, num_classes, reg_max, dim=64,
                 class_ids=(1, 2), neighbor_count=4, neighbor_iou=0.5,
                 common_limit=0.5, contrast_limit=1.5):
        super().__init__()
        self.class_ids = tuple(class_ids)
        if (len(self.class_ids) != 2
                or any(type(c) is not int for c in self.class_ids)
                or len(set(self.class_ids)) != 2
                or any(c < 0 or c >= num_classes for c in self.class_ids)):
            raise ValueError("RW calibration needs two distinct valid integer class IDs")
        if any(type(v) is not int or v <= 0 for v in (hidden_dim, dim, neighbor_count, reg_max)):
            raise ValueError("RW calibration dimensions, bins and neighbor count must be positive integers")
        if not math.isfinite(neighbor_iou) or not 0 < neighbor_iou <= 1:
            raise ValueError("rw_calibration_neighbor_iou must be in (0, 1]")
        if any(not math.isfinite(v) or v <= 0 for v in (common_limit, contrast_limit)):
            raise ValueError("RW calibration correction limits must be finite and positive")
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.neighbor_count = neighbor_count
        self.neighbor_iou = neighbor_iou
        self.common_limit = common_limit
        self.contrast_limit = contrast_limit
        self.project = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, dim), nn.SiLU())
        # Own/context feature, own/context class scores, four entropies + peaks.
        self.fuse = nn.Sequential(nn.Linear(2 * dim + 2 * num_classes + 8, dim),
                                  nn.SiLU(), nn.Linear(dim, 2))
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    @torch.no_grad()
    def neighbor_weights(self, boxes, logits):
        """Within-image, self-excluding IoU neighbors. Empty rows stay zero."""
        boxes = boxes.detach().float()
        center, half = boxes[..., :2], boxes[..., 2:] / 2
        left, right = center - half, center + half
        intersection = (torch.minimum(right[:, :, None], right[:, None, :])
                        - torch.maximum(left[:, :, None], left[:, None, :])).clamp_min(0).prod(-1)
        area = (right - left).clamp_min(0).prod(-1)
        iou = intersection / (area[:, :, None] + area[:, None, :] - intersection).clamp_min(1e-8)
        queries = boxes.shape[1]
        self_mask = torch.eye(queries, dtype=torch.bool, device=boxes.device).unsqueeze(0)
        eligible = (iou >= self.neighbor_iou) & ~self_mask
        count = min(self.neighbor_count, max(queries - 1, 0))
        if count == 0:
            return torch.zeros_like(iou)
        top = iou.masked_fill(~eligible, -1).topk(count, dim=-1).indices
        keep = torch.zeros_like(eligible).scatter_(-1, top, True) & eligible
        importance = logits.detach().float()[..., list(self.class_ids)].sigmoid().amax(-1)
        weights = iou * keep * importance[:, None, :]
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def quality_features(self, corners):
        probabilities = corners.detach().float().reshape(*corners.shape[:2], 4, self.reg_max + 1).softmax(-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)
        entropy = entropy / math.log(self.reg_max + 1)
        return torch.cat((entropy, probabilities.amax(-1)), dim=-1)

    def forward(self, query, boxes, logits, corners):
        if (query.ndim != 3 or boxes.shape != (*query.shape[:2], 4)
                or logits.shape != (*query.shape[:2], self.num_classes)
                or corners.shape != (*query.shape[:2], 4 * (self.reg_max + 1))):
            raise ValueError("RW calibration inputs must use the same ordinary final queries")
        projected = self.project(query.detach())
        weights = self.neighbor_weights(boxes, logits)
        context = torch.bmm(weights.to(projected.dtype), projected)
        scores = logits.detach().float().sigmoid()
        context_scores = torch.bmm(weights, scores)
        evidence = torch.cat((projected, context, scores.to(projected.dtype),
                              context_scores.to(projected.dtype),
                              self.quality_features(corners).to(projected.dtype)), dim=-1)
        raw = self.fuse(evidence).float()
        common = self.common_limit * raw[..., 0].tanh()
        contrast = self.contrast_limit * raw[..., 1].tanh()
        correction = torch.stack((common + contrast, common - contrast), dim=-1)
        return logits.detach().float()[..., list(self.class_ids)] + correction

    def refine_logits(self, logits, calibrated_pair):
        """Keep every other class exactly unchanged; no softmax substitution."""
        if calibrated_pair.shape != (*logits.shape[:2], 2):
            raise ValueError("Calibrated read/write logits must have shape [B,Q,2]")
        refined = logits.clone()
        refined[..., list(self.class_ids)] = calibrated_pair.to(logits.dtype)
        return refined
