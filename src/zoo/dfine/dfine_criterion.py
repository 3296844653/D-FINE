"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
import math

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ...core import register
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .dfine_utils import bbox2distance


def shape_iou_loss(pred_boxes, target_boxes, scale=0.0, eps=1e-7):
    """Shape-IoU loss for aligned ``cxcywh`` box pairs.

    Adapted from the official implementation:
    https://github.com/malagoutou/Shape-IoU/blob/main/shapeiou.py
    """
    if pred_boxes.shape != target_boxes.shape or pred_boxes.shape[-1] != 4:
        raise ValueError("pred_boxes and target_boxes must have the same shape [..., 4]")
    if pred_boxes.numel() == 0:
        return pred_boxes.new_zeros(pred_boxes.shape[:-1])

    # Compute the geometric terms in fp32 under AMP for numerical stability.
    pred = pred_boxes.float() if pred_boxes.dtype in (torch.float16, torch.bfloat16) else pred_boxes
    target = (
        target_boxes.float()
        if target_boxes.dtype in (torch.float16, torch.bfloat16)
        else target_boxes
    )

    pred_x, pred_y, pred_w, pred_h = pred.unbind(-1)
    target_x, target_y, target_w, target_h = target.unbind(-1)
    pred_w = pred_w.clamp_min(eps)
    pred_h = pred_h.clamp_min(eps)
    target_w = target_w.clamp_min(eps)
    target_h = target_h.clamp_min(eps)

    pred_x1, pred_x2 = pred_x - pred_w / 2, pred_x + pred_w / 2
    pred_y1, pred_y2 = pred_y - pred_h / 2, pred_y + pred_h / 2
    target_x1, target_x2 = target_x - target_w / 2, target_x + target_w / 2
    target_y1, target_y2 = target_y - target_h / 2, target_y + target_h / 2

    inter_w = (torch.minimum(pred_x2, target_x2) - torch.maximum(pred_x1, target_x1)).clamp_min(0)
    inter_h = (torch.minimum(pred_y2, target_y2) - torch.maximum(pred_y1, target_y1)).clamp_min(0)
    intersection = inter_w * inter_h
    union = (pred_w * pred_h + target_w * target_h - intersection).clamp_min(eps)
    iou = intersection / union

    target_w_scaled = target_w.pow(scale)
    target_h_scaled = target_h.pow(scale)
    shape_denominator = (target_w_scaled + target_h_scaled).clamp_min(eps)
    width_weight = 2 * target_w_scaled / shape_denominator
    height_weight = 2 * target_h_scaled / shape_denominator

    convex_w = (torch.maximum(pred_x2, target_x2) - torch.minimum(pred_x1, target_x1)).clamp_min(eps)
    convex_h = (torch.maximum(pred_y2, target_y2) - torch.minimum(pred_y1, target_y1)).clamp_min(eps)
    convex_diagonal = (convex_w.square() + convex_h.square()).clamp_min(eps)
    center_distance = (
        height_weight * (pred_x - target_x).square()
        + width_weight * (pred_y - target_y).square()
    ) / convex_diagonal

    width_difference = (
        height_weight
        * (pred_w - target_w).abs()
        / torch.maximum(pred_w, target_w).clamp_min(eps)
    )
    height_difference = (
        width_weight
        * (pred_h - target_h).abs()
        / torch.maximum(pred_h, target_h).clamp_min(eps)
    )
    shape_cost = (1 - torch.exp(-width_difference)).pow(4) + (
        1 - torch.exp(-height_difference)
    ).pow(4)

    shape_iou = iou - center_distance - 0.5 * shape_cost
    return 1 - shape_iou


@register()
class DFINECriterion(nn.Module):
    """This class computes the loss for D-FINE."""

    __share__ = [
        "num_classes",
    ]
    __inject__ = [
        "matcher",
    ]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        use_class_margin=False,
        class_margin=0.2,
        class_margin_weight=0.1,
        class_margin_pairs=None,
        use_query_validity=False,
        query_validity_weight=0.1,
        use_query_objectness=False,
        query_objectness_weight=0.1,
        query_objectness_positive_iou=0.5,
        query_objectness_negative_iou=0.1,
        query_objectness_negative_ratio=3,
        query_objectness_min_negatives=32,
        use_shape_iou=False,
        shape_iou_scale=0.0,
        shape_iou_eps=1e-7,
        use_class_balanced_vfl=False,
        vfl_positive_class_weights=None,
        use_pairwise_ce=False,
        pairwise_ce_classes=(1, 2),
        pairwise_ce_weight=0.1,
        use_rw_isolated_specialist=False,
        rw_specialist_class_ids=(1, 2),
        rw_specialist_weight=0.1,
        rw_specialist_min_iou=0.5,
        use_rw_query_calibration=False,
        rw_calibration_class_ids=(1, 2),
        rw_calibration_weight=0.1,
        rw_calibration_positive_iou=0.5,
        rw_calibration_negative_iou=0.3,
        rw_calibration_negative_ratio=3,
        rw_calibration_min_negatives=16,
    ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in D-FINE.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.use_rw_query_calibration = bool(use_rw_query_calibration)
        self.rw_calibration_class_ids = tuple(rw_calibration_class_ids)
        self.rw_calibration_weight = float(rw_calibration_weight)
        self.rw_calibration_positive_iou = float(rw_calibration_positive_iou)
        self.rw_calibration_negative_iou = float(rw_calibration_negative_iou)
        self.rw_calibration_negative_ratio = rw_calibration_negative_ratio
        self.rw_calibration_min_negatives = rw_calibration_min_negatives
        if self.use_rw_query_calibration:
            ids = self.rw_calibration_class_ids
            if (len(ids) != 2 or any(type(c) is not int for c in ids)
                    or len(set(ids)) != 2 or any(c < 0 or c >= num_classes for c in ids)):
                raise ValueError("rw_calibration_class_ids must contain two distinct valid integer IDs")
            if not math.isfinite(self.rw_calibration_weight) or self.rw_calibration_weight <= 0:
                raise ValueError("rw_calibration_weight must be finite and positive")
            if not 0 <= self.rw_calibration_negative_iou < self.rw_calibration_positive_iou <= 1:
                raise ValueError("RW calibration requires 0 <= negative IoU < positive IoU <= 1")
            if any(type(v) is not int or v < 1 for v in
                   (rw_calibration_negative_ratio, rw_calibration_min_negatives)):
                raise ValueError("RW calibration negative sampling settings must be positive integers")
            if any((use_pairwise_ce, use_class_margin, use_class_balanced_vfl,
                    use_query_validity, use_query_objectness, use_rw_isolated_specialist)):
                raise ValueError("RW query calibration must be an independent classification experiment")
        self.use_rw_isolated_specialist = bool(use_rw_isolated_specialist)
        self.rw_specialist_class_ids = tuple(rw_specialist_class_ids)
        self.rw_specialist_weight = float(rw_specialist_weight)
        self.rw_specialist_min_iou = float(rw_specialist_min_iou)
        if self.use_rw_isolated_specialist:
            if (len(self.rw_specialist_class_ids) != 2
                or any(type(c) is not int for c in self.rw_specialist_class_ids)
                or len(set(self.rw_specialist_class_ids)) != 2
                or any(c < 0 or c >= num_classes for c in self.rw_specialist_class_ids)):
                raise ValueError("rw_specialist_class_ids must contain two distinct valid integer IDs")
            if not math.isfinite(self.rw_specialist_weight) or self.rw_specialist_weight <= 0:
                raise ValueError("rw_specialist_weight must be finite and positive")
            if not math.isfinite(self.rw_specialist_min_iou) or not 0 <= self.rw_specialist_min_iou <= 1:
                raise ValueError("rw_specialist_min_iou must be in [0, 1]")
            if use_pairwise_ce or use_class_margin or use_class_balanced_vfl or use_query_validity or use_query_objectness:
                raise ValueError("RW isolated specialist must be an independent classification experiment")
        # Final matched queries only: conditional discrimination inside a pair.
        # The detector's sigmoid scores, VFL and matching cost remain unchanged.
        self.use_pairwise_ce = bool(use_pairwise_ce)
        self.pairwise_ce_classes = tuple(pairwise_ce_classes)
        self.pairwise_ce_weight = float(pairwise_ce_weight)
        if self.use_pairwise_ce:
            if (len(self.pairwise_ce_classes) != 2
                or any(type(c) is not int for c in self.pairwise_ce_classes)
                or len(set(self.pairwise_ce_classes)) != 2
                or any(c < 0 or c >= num_classes for c in self.pairwise_ce_classes)):
                raise ValueError("pairwise_ce_classes must contain two distinct valid integer class IDs")
            if not math.isfinite(self.pairwise_ce_weight) or self.pairwise_ce_weight <= 0:
                raise ValueError("pairwise_ce_weight must be finite and positive")
            if use_class_margin or use_class_balanced_vfl or use_query_validity or use_query_objectness:
                raise ValueError("Pairwise CE must be tested independently of other classification losses")
        # Optional positive-only VFL weighting. Category order follows model IDs.
        # No learnable parameters or buffers: existing checkpoints stay compatible.
        self.use_class_balanced_vfl = bool(use_class_balanced_vfl)
        if vfl_positive_class_weights is None:
            vfl_positive_class_weights = [1.0] * num_classes
        weights = torch.as_tensor(vfl_positive_class_weights, dtype=torch.float32)
        if weights.ndim != 1 or weights.numel() != num_classes:
            raise ValueError("vfl_positive_class_weights must have num_classes entries")
        if not torch.isfinite(weights).all() or not (weights > 0).all():
            raise ValueError("VFL positive class weights must be finite and positive")
        self.vfl_positive_class_weights = tuple(float(w) for w in weights)
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.use_shape_iou = bool(use_shape_iou)
        self.shape_iou_scale = float(shape_iou_scale)
        self.shape_iou_eps = float(shape_iou_eps)
        if self.shape_iou_scale < 0:
            raise ValueError("shape_iou_scale must be non-negative")
        if self.shape_iou_eps <= 0:
            raise ValueError("shape_iou_eps must be positive")
        self.num_pos, self.num_neg = None, None
        # Optional final-layer ranking constraint for matched detection queries.
        # It complements VFL's absolute IoU-aware targets without replacing VFL.
        self.use_class_margin = use_class_margin
        self.class_margin = class_margin
        self.class_margin_weight = class_margin_weight
        if class_margin < 0 or class_margin_weight < 0:
            raise ValueError("Class margin and its weight must be non-negative")
        self.class_margin_pairs = None
        if class_margin_pairs is not None:
            normalized_pairs = []
            for pair in class_margin_pairs:
                if len(pair) != 2:
                    raise ValueError("Each class-margin pair must contain exactly two class IDs")
                first, second = (int(pair[0]), int(pair[1]))
                if first == second:
                    raise ValueError("A class-margin pair must contain two different classes")
                if not (0 <= first < self.num_classes and 0 <= second < self.num_classes):
                    raise ValueError("Class-margin pair contains an out-of-range class ID")
                normalized_pairs.append((first, second))
            if not normalized_pairs:
                raise ValueError("class_margin_pairs cannot be empty when provided")
            self.class_margin_pairs = tuple(normalized_pairs)
        self.use_query_validity = use_query_validity
        self.query_validity_weight = query_validity_weight
        if query_validity_weight < 0:
            raise ValueError("Query validity loss weight must be non-negative")
        self.use_query_objectness = use_query_objectness
        self.query_objectness_weight = query_objectness_weight
        self.query_objectness_positive_iou = query_objectness_positive_iou
        self.query_objectness_negative_iou = query_objectness_negative_iou
        self.query_objectness_negative_ratio = query_objectness_negative_ratio
        self.query_objectness_min_negatives = query_objectness_min_negatives
        if use_query_objectness and (use_query_validity or use_class_margin):
            raise ValueError("Query objectness must be tested separately from other classification losses")
        if not 0 <= query_objectness_negative_iou < query_objectness_positive_iou <= 1:
            raise ValueError("Objectness requires 0 <= negative IoU < positive IoU <= 1")
        if query_objectness_weight < 0 or query_objectness_negative_ratio <= 0 or query_objectness_min_negatives < 1:
            raise ValueError("Invalid query objectness loss weight or negative sampling settings")

    def loss_query_objectness(self, outputs, targets):
        """Detached-box geometric targets, final ordinary queries only.

        Any query with max GT IoU >= positive threshold is positive, including
        duplicates. Queries <= negative threshold are negative candidates;
        intermediate overlaps are ignored. Hard negatives are chosen by the
        ORIGINAL detached classification score, never by validation error lists.
        Group-balanced BCE avoids domination by thousands of easy negatives.
        This is task-scope validity relative to annotations, not generic personness.
        """
        logits = outputs["query_objectness_logits"].squeeze(-1).float()
        if logits.shape != outputs["pred_logits"].shape[:2]:
            raise ValueError("Objectness shape differs from ordinary detection queries")
        positive_values, negative_values = [], []
        with torch.no_grad():
            boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"].detach().float())
            scores = outputs["pred_logits"].detach().float().sigmoid().amax(-1)
        for b, target in enumerate(targets):
            with torch.no_grad():
                gt = box_cxcywh_to_xyxy(target["boxes"].detach().float())
                max_iou = box_iou(boxes[b], gt)[0].amax(-1) if len(gt) else scores[b].new_zeros(scores.shape[1])
                positive = max_iou >= self.query_objectness_positive_iou
                negatives = torch.where(max_iou <= self.query_objectness_negative_iou)[0]
                count = min(len(negatives), max(self.query_objectness_min_negatives,
                    int(self.query_objectness_negative_ratio * int(positive.sum()))))
                selected = negatives[scores[b, negatives].topk(count).indices] if count else negatives
            positive_values.append(logits[b, positive])
            negative_values.append(logits[b, selected])
        terms = []
        for values, label in [(positive_values, 1.0), (negative_values, 0.0)]:
            values = torch.cat(values) if values else logits.reshape(-1)[:0]
            if values.numel():
                terms.append(F.binary_cross_entropy_with_logits(values, torch.full_like(values, label)))
        return torch.stack(terms).mean() if terms else logits.sum() * 0

    @staticmethod
    def loss_query_validity(outputs, indices):
        """Balanced matched/unmatched binary supervision for final queries only.

        Hungarian-matched queries are positive; all other ordinary queries are
        negative. Positive and negative means are balanced per batch so the
        numerous easy negatives cannot overwhelm the sparse positives.
        """
        logits = outputs["query_validity_logits"].squeeze(-1)
        if logits.shape != outputs["pred_logits"].shape[:2]:
            raise ValueError("Query validity logits do not match detection queries")
        target = torch.zeros_like(logits)
        for batch_idx, (query_indices, _) in enumerate(indices):
            target[batch_idx, query_indices.to(device=logits.device)] = 1
        per_query = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        positive = target.bool()
        positive_loss = per_query[positive].mean() if positive.any() else logits.sum() * 0
        negative_loss = per_query[~positive].mean() if (~positive).any() else logits.sum() * 0
        return 0.5 * (positive_loss + negative_loss)

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits, target, self.alpha, self.gamma, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {"loss_focal": loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        # Only matched GT-class entries get higher weight, including aux / DN.
        # Wrong-class entries and unmatched queries keep their original weights.
        # Class-agnostic encoder outputs have one channel and are left unchanged.
        if self.use_class_balanced_vfl and src_logits.shape[-1] == len(self.vfl_positive_class_weights):
            class_weights = src_logits.new_tensor(self.vfl_positive_class_weights)
            weight = weight + target_score * (class_weights - 1)

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    def loss_labels_class_margin(self, outputs, targets, indices, num_boxes):
        """Penalize a matched query only when a wrong class rivals its GT class.

        Hungarian matching and VFL remain unchanged. This loss is applied only
        to the final decoder output, not auxiliary, denoising, or encoder heads.
        When ``class_margin_pairs`` is configured, only those symmetric
        confusion relations participate; otherwise the legacy all-class
        strongest-competitor behavior is preserved.
        """
        src_logits = outputs["pred_logits"]
        if self.num_classes < 2 or not any(len(src) for src, _ in indices):
            return src_logits.sum() * 0

        idx = self._get_src_permutation_idx(indices)
        matched_logits = src_logits[idx].float()
        gt_classes = torch.cat(
            [target["labels"][gt] for target, (_, gt) in zip(targets, indices)]
        ).to(matched_logits.device)
        correct_logits = matched_logits.gather(1, gt_classes[:, None]).squeeze(1)

        if self.class_margin_pairs is None:
            wrong_mask = F.one_hot(gt_classes, num_classes=self.num_classes).bool()
            highest_wrong_logits = matched_logits.masked_fill(wrong_mask, -torch.inf).max(dim=1).values
            active = torch.ones_like(gt_classes, dtype=torch.bool)
        else:
            candidate_mask = torch.zeros_like(matched_logits, dtype=torch.bool)
            for first, second in self.class_margin_pairs:
                candidate_mask[gt_classes == first, second] = True
                candidate_mask[gt_classes == second, first] = True
            active = candidate_mask.any(dim=1)
            if not active.any():
                return matched_logits.sum() * 0
            highest_wrong_logits = matched_logits.masked_fill(
                ~candidate_mask, -torch.inf
            ).max(dim=1).values

        margin_loss = F.relu(
            highest_wrong_logits[active] - correct_logits[active] + self.class_margin
        )
        # Normalize by the same global matched-box count as the legacy margin
        # experiment so restricting the confusion set is the only main change.
        return margin_loss.sum() / num_boxes

    def loss_labels_pairwise_ce(self, outputs, targets, indices, num_boxes):
        """Conditional two-class CE, not a replacement for detection confidence.

        Uses final-layer Hungarian positives whose GT belongs to the pair.
        Background, other classes, auxiliary and DN queries do not participate.
        Global GT normalization follows VFL; no class frequency weighting or
        ground-truth information is introduced at inference time.
        """
        logits = outputs["pred_logits"]
        if not any(len(src) for src, _ in indices):
            return logits.sum() * 0
        matched = logits[self._get_src_permutation_idx(indices)].float()
        labels = torch.cat([
            target["labels"][gt] for target, (_, gt) in zip(targets, indices)
        ]).to(matched.device)
        first, second = self.pairwise_ce_classes
        active = (labels == first) | (labels == second)
        if not active.any():
            return matched.sum() * 0
        pair_logits = matched[active][:, [first, second]]
        pair_targets = (labels[active] == second).long()
        return F.cross_entropy(pair_logits, pair_targets, reduction="sum") / num_boxes

    def loss_rw_isolated_specialist(self, outputs, targets, indices, num_boxes):
        """Only clean original Hungarian pair matches train the detached expert."""
        if "rw_specialist_logits" not in outputs:
            raise ValueError("RW specialist loss requires its enabled decoder branch")
        if tuple(outputs.get("rw_specialist_class_ids", ())) != self.rw_specialist_class_ids:
            raise ValueError("Decoder/criterion RW specialist class IDs do not match")
        logits = outputs["rw_specialist_logits"]
        if logits.shape != (*outputs["pred_logits"].shape[:2], 2):
            raise ValueError("RW specialist logits must have shape [B,Q,2]")
        if not any(len(src) for src, _ in indices):
            return logits.sum() * 0
        idx = self._get_src_permutation_idx(indices)
        labels = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
        first, second = self.rw_specialist_class_ids
        with torch.no_grad():
            boxes = outputs["pred_boxes"][idx].float()
            gt_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)]).float()
            ious = torch.diag(box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(gt_boxes))[0])
            active = ((labels == first) | (labels == second)) & (ious >= self.rw_specialist_min_iou)
        if not active.any():
            return logits.sum() * 0
        return F.cross_entropy(
            logits[idx][active].float(), (labels[active] == second).long(), reduction="sum"
        ) / num_boxes

    def loss_rw_query_calibration(self, outputs, targets, indices, num_boxes):
        """Quality targets for clean ONE-TO-ONE matches, not every overlap.

        Unmatched low-overlap candidates and clean matches of other classes are
        hard-negative candidates. Duplicates, gray overlaps and poor matches are
        ignored. Labels express dataset scope; low IoU does not prove background.
        """
        if "rw_calibration_logits" not in outputs:
            raise ValueError("RW calibration loss requires its enabled decoder branch")
        if tuple(outputs.get("rw_calibration_class_ids", ())) != self.rw_calibration_class_ids:
            raise ValueError("Decoder/criterion RW calibration class IDs do not match")
        logits = outputs["rw_calibration_logits"].float()
        if logits.shape != (*outputs["pred_logits"].shape[:2], 2):
            raise ValueError("RW calibration logits must have shape [B,Q,2]")
        first, second = self.rw_calibration_class_ids
        total = logits.sum() * 0
        with torch.no_grad():
            boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"].detach().float())
            scores = outputs["pred_logits"].detach().float()[..., [first, second]].sigmoid().amax(-1)
        for b, (target, (source, gt_indices)) in enumerate(zip(targets, indices)):
            with torch.no_grad():
                # SciPy HungarianMatcher returns CPU index tensors even when
                # the model/GT are on CUDA. Boolean masks below are on the
                # logits device, so normalize the local copies BEFORE indexing
                # source with a mask. Do not change the original match list.
                source = source.to(device=logits.device, dtype=torch.long)
                gt_indices = gt_indices.to(device=logits.device, dtype=torch.long)
                gt_boxes = box_cxcywh_to_xyxy(
                    target["boxes"].detach().to(device=logits.device, dtype=torch.float32)
                )
                overlaps = box_iou(boxes[b], gt_boxes)[0]
                max_iou = overlaps.amax(-1) if len(gt_boxes) else scores[b].new_zeros(scores.shape[1])
                matched = torch.zeros_like(scores[b], dtype=torch.bool)
                matched[source] = True
                negatives = ~matched & (max_iou <= self.rw_calibration_negative_iou)
                labels = target["labels"].to(device=logits.device, dtype=torch.long)[gt_indices]
                matched_iou = overlaps[source, gt_indices]
                clean = matched_iou >= self.rw_calibration_positive_iou
                is_pair = (labels == first) | (labels == second)
                positives = source[clean & is_pair]
                pair_targets = logits.new_zeros((len(positives), 2))
                pair_targets[torch.arange(len(positives), device=logits.device),
                             (labels[clean & is_pair] == second).long()] = matched_iou[clean & is_pair]
                negatives[source[clean & ~is_pair]] = True
                negative_indices = torch.where(negatives)[0]
                count = min(len(negative_indices), max(self.rw_calibration_min_negatives,
                            self.rw_calibration_negative_ratio * len(positives)))
                selected = negative_indices[scores[b, negative_indices].topk(count).indices] if count else negative_indices
            if len(positives):
                total = total + F.binary_cross_entropy_with_logits(
                    logits[b, positives], pair_targets, reduction="none"
                ).mean(-1).sum()
            if len(selected):
                # Group balancing: easy negatives cannot swamp the positives.
                total = total + max(len(positives), 1) * F.binary_cross_entropy_with_logits(
                    logits[b, selected], torch.zeros_like(logits[b, selected]), reduction="mean"
                )
        return total / num_boxes

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        if self.use_shape_iou:
            loss_giou = shape_iou_loss(
                src_boxes,
                target_boxes,
                scale=self.shape_iou_scale,
                eps=self.shape_iou_eps,
            )
        else:
            loss_giou = 1 - torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes),
                    box_cxcywh_to_xyxy(target_boxes),
                )
            )
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss
        and Decoupled Distillation Focal (DDF) Loss."""

        losses = {}
        if "pred_corners" in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs["pred_corners"][idx].reshape(-1, (self.reg_max + 1))
            ref_points = outputs["ref_points"][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and "is_dn" in outputs:
                    self.fgl_targets_dn = bbox2distance(
                        ref_points,
                        box_cxcywh_to_xyxy(target_boxes),
                        self.reg_max,
                        outputs["reg_scale"],
                        outputs["up"],
                    )
                if self.fgl_targets is None and "is_dn" not in outputs:
                    self.fgl_targets = bbox2distance(
                        ref_points,
                        box_cxcywh_to_xyxy(target_boxes),
                        self.reg_max,
                        outputs["reg_scale"],
                        outputs["up"],
                    )

            target_corners, weight_right, weight_left = (
                self.fgl_targets_dn if "is_dn" in outputs else self.fgl_targets
            )

            ious = torch.diag(
                box_iou(
                    box_cxcywh_to_xyxy(outputs["pred_boxes"][idx]), box_cxcywh_to_xyxy(target_boxes)
                )[0]
            )
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses["loss_fgl"] = self.unimodal_distribution_focal_loss(
                pred_corners,
                target_corners,
                weight_right,
                weight_left,
                weight_targets,
                avg_factor=num_boxes,
            )

            if "teacher_corners" in outputs:
                pred_corners = outputs["pred_corners"].reshape(-1, (self.reg_max + 1))
                target_corners = outputs["teacher_corners"].reshape(-1, (self.reg_max + 1))
                if torch.equal(pred_corners, target_corners):
                    losses["loss_ddf"] = pred_corners.sum() * 0
                else:
                    weight_targets_local = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(
                        weight_targets_local.dtype
                    )
                    weight_targets_local = (
                        weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()
                    )

                    loss_match_local = (
                        weight_targets_local
                        * (T**2)
                        * (
                            nn.KLDivLoss(reduction="none")(
                                F.log_softmax(pred_corners / T, dim=1),
                                F.softmax(target_corners.detach() / T, dim=1),
                            )
                        ).sum(-1)
                    )
                    if "is_dn" not in outputs:
                        batch_scale = (
                            8 / outputs["pred_boxes"].shape[0]
                        )  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (
                            (mask.sum() * batch_scale) ** 0.5,
                            ((~mask).sum() * batch_scale) ** 0.5,
                        )
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses["loss_ddf"] = (
                        loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg
                    ) / (self.num_pos + self.num_neg)

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers."""
        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
            "local": self.loss_local,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}
        if ("rw_calibration_logits" in outputs_without_aux) != self.use_rw_query_calibration:
            raise ValueError("Decoder and criterion must enable RW query calibration together")

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)["indices"]
        self._clear_cache()

        # Get the matching union set across all decoder layers.
        if "aux_outputs" in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            for i, aux_outputs in enumerate(outputs["aux_outputs"] + [outputs["pre_outputs"]]):
                indices_aux = self.matcher(aux_outputs, targets)["indices"]
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                indices_enc = self.matcher(aux_outputs, targets)["indices"]
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor(
                [num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device
            )
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            assert "aux_outputs" in outputs, ""

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            indices_in = indices_go if loss in ["boxes", "local"] else indices
            num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        if self.use_pairwise_ce:
            losses["loss_pairwise_ce"] = self.pairwise_ce_weight * self.loss_labels_pairwise_ce(
                outputs_without_aux, targets, indices, num_boxes
            )

        if self.use_rw_isolated_specialist:
            losses["loss_rw_specialist"] = self.rw_specialist_weight * self.loss_rw_isolated_specialist(
                outputs_without_aux, targets, indices, num_boxes
            )

        if self.use_rw_query_calibration:
            losses["loss_rw_query_calibration"] = self.rw_calibration_weight * self.loss_rw_query_calibration(
                outputs_without_aux, targets, indices, num_boxes
            )

        if self.use_class_margin:
            losses["loss_class_margin"] = self.class_margin_weight * self.loss_labels_class_margin(
                outputs_without_aux, targets, indices, num_boxes
            )

        if self.use_query_validity:
            if "query_validity_logits" not in outputs_without_aux:
                raise ValueError("Query validity loss requires query_validity_logits")
            losses["loss_query_validity"] = self.query_validity_weight * self.loss_query_validity(
                outputs_without_aux, indices
            )
        if self.use_query_objectness:
            if "query_objectness_logits" not in outputs_without_aux:
                raise ValueError("Objectness loss requires the enabled decoder branch")
            losses["loss_query_objectness"] = self.query_objectness_weight * self.loss_query_objectness(
                outputs_without_aux, targets
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
                for loss in self.losses:
                    indices_in = indices_go if loss in ["boxes", "local"] else cached_indices[i]
                    num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_in, num_boxes_in, **meta
                    )

                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer.
        if "pre_outputs" in outputs:
            aux_outputs = outputs["pre_outputs"]
            for loss in self.losses:
                indices_in = indices_go if loss in ["boxes", "local"] else cached_indices[-1]
                num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                l_dict = {
                    k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                }
                l_dict = {k + "_pre": v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if "enc_aux_outputs" in outputs:
            assert "enc_meta" in outputs, ""
            class_agnostic = outputs["enc_meta"]["class_agnostic"]
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t["labels"] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                for loss in self.losses:
                    indices_in = indices_go if loss == "boxes" else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if loss == "boxes" else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(
                        loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses. For dfine
        if "dn_outputs" in outputs:
            assert "dn_meta" in outputs, ""
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]
            dn_num_boxes = dn_num_boxes if dn_num_boxes > 0 else 1

            for i, aux_outputs in enumerate(outputs["dn_outputs"]):
                aux_outputs["is_dn"] = True
                aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer.
            if "dn_pre_outputs" in outputs:
                aux_outputs = outputs["dn_pre_outputs"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + "_dn_pre": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # For debugging Objects365 pre-train.
        losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs["pred_boxes"][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == "iou":
            iou, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)
            )
            iou = torch.diag(iou)
        elif self.boxes_weight_format == "giou":
            iou = torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)
                )
            )
        else:
            raise AttributeError()

        if loss in ("boxes",):
            meta = {"boxes_weight": iou}
        elif loss in ("vfl",):
            meta = {"values": iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices"""
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t["labels"]) for t in targets]
        device = targets[0]["labels"].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append(
                    (
                        torch.zeros(0, dtype=torch.int64, device=device),
                        torch.zeros(0, dtype=torch.int64, device=device),
                    )
                )

        return dn_match_indices

    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)

    def unimodal_distribution_focal_loss(
        self, pred, label, weight_right, weight_left, weight=None, reduction="sum", avg_factor=None
    ):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(
            -1
        ) + F.cross_entropy(pred, dis_right, reduction="none") * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs["aux_outputs"]) + 1 if "aux_outputs" in outputs else 1
        step = 0.5 / (num_layers - 1)
        opt_list = [0.5 + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list
