"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ...core import register
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .dfine_utils import bbox2distance


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
        use_o2m_aux=False,
        o2m_topk=3,
        o2m_loss_weight=0.25,
        o2m_losses=None,
        o2m_matcher_type="cost",
        o2m_quality_gamma=0.25,
        o2m_min_pos=1,
        use_class_aware_vfl=False,
        vfl_class_counts=None,
        vfl_class_weight_power=0.25,
        vfl_class_weight_max=2.0,
        rank_gcl_gamma=2.0,
        align_quality_alpha=0.25,
        align_gamma=2.0,
        align_min_quality=0.01,
        eqlv2_gamma=12.0,
        eqlv2_mu=0.8,
        eqlv2_alpha=4.0,
        use_opl=False,
        opl_gamma=0.5,
        opl_loss_weight=0.1,
        use_ccp_loss=False,
        ccp_feature_dim=256,
        ccp_loss_weight=0.05,
        ccp_temperature=0.1,
        ccp_momentum=0.9,
        ccp_confusion_weight=1.0,
        use_lahns=False,
        lahns_loss_weight=0.05,
        lahns_score_threshold=0.4,
        lahns_iou_power=2.0,
        lahns_score_power=2.0,
        lahns_topk=30,
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
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.use_o2m_aux = use_o2m_aux
        self.o2m_topk = max(int(o2m_topk), 1)
        self.o2m_loss_weight = float(o2m_loss_weight)
        self.o2m_losses = o2m_losses if o2m_losses is not None else ["vfl", "boxes"]
        assert o2m_matcher_type in ("cost", "quality")
        self.o2m_matcher_type = o2m_matcher_type
        self.o2m_quality_gamma = float(o2m_quality_gamma)
        self.o2m_min_pos = max(int(o2m_min_pos), 1)
        self.use_class_aware_vfl = bool(use_class_aware_vfl)
        if self.use_class_aware_vfl:
            assert vfl_class_counts is not None, (
                "vfl_class_counts is required when use_class_aware_vfl=True"
            )
            assert len(vfl_class_counts) == num_classes, (
                "vfl_class_counts must contain one count per category"
            )
            assert all(float(count) > 0 for count in vfl_class_counts), (
                "all vfl_class_counts must be positive"
            )
            assert float(vfl_class_weight_power) >= 0, (
                "vfl_class_weight_power must be non-negative"
            )
            assert float(vfl_class_weight_max) >= 1, (
                "vfl_class_weight_max must be >= 1"
            )
            class_counts = torch.as_tensor(vfl_class_counts, dtype=torch.float32)
            class_weights = (class_counts.max() / class_counts).pow(
                float(vfl_class_weight_power)
            )
            class_weights = class_weights.clamp(max=float(vfl_class_weight_max))
        else:
            class_weights = torch.ones(num_classes, dtype=torch.float32)



        self.register_buffer("vfl_class_weights", class_weights, persistent=False)




        self.rank_gcl_gamma = float(rank_gcl_gamma)
        assert self.rank_gcl_gamma >= 0




        self.align_quality_alpha = float(align_quality_alpha)
        self.align_gamma = float(align_gamma)
        self.align_min_quality = float(align_min_quality)
        assert 0 <= self.align_quality_alpha <= 1
        assert self.align_gamma >= 0
        assert 0 <= self.align_min_quality <= 1

        self.eqlv2_gamma = float(eqlv2_gamma)
        self.eqlv2_mu = float(eqlv2_mu)
        self.eqlv2_alpha = float(eqlv2_alpha)
        assert self.eqlv2_gamma > 0
        assert 0 <= self.eqlv2_mu <= 1
        assert self.eqlv2_alpha >= 0
        self.register_buffer("eqlv2_pos_grad", torch.zeros(num_classes))
        self.register_buffer("eqlv2_neg_grad", torch.zeros(num_classes))
        self.register_buffer("eqlv2_pos_neg", torch.full((num_classes,), 100.0))

        self.use_opl = bool(use_opl)
        self.opl_gamma = float(opl_gamma)
        self.opl_loss_weight = float(opl_loss_weight)
        assert self.opl_gamma >= 0
        assert self.opl_loss_weight >= 0




        self.use_ccp_loss = bool(use_ccp_loss)
        self.ccp_feature_dim = int(ccp_feature_dim)
        self.ccp_loss_weight = float(ccp_loss_weight)
        self.ccp_temperature = float(ccp_temperature)
        self.ccp_momentum = float(ccp_momentum)
        self.ccp_confusion_weight = float(ccp_confusion_weight)
        assert self.ccp_feature_dim > 0
        assert self.ccp_loss_weight >= 0
        assert self.ccp_temperature > 0
        assert 0 <= self.ccp_momentum < 1
        assert self.ccp_confusion_weight >= 0
        self.register_buffer(
            "ccp_prototypes",
            torch.zeros(num_classes, self.ccp_feature_dim),
            persistent=False,
        )
        self.register_buffer(
            "ccp_prototype_counts",
            torch.zeros(num_classes),
            persistent=False,
        )





        self.use_lahns = bool(use_lahns)
        self.lahns_loss_weight = float(lahns_loss_weight)
        self.lahns_score_threshold = float(lahns_score_threshold)
        self.lahns_iou_power = float(lahns_iou_power)
        self.lahns_score_power = float(lahns_score_power)
        self.lahns_topk = int(lahns_topk)
        assert self.lahns_loss_weight >= 0
        assert 0 <= self.lahns_score_threshold <= 1
        assert self.lahns_iou_power >= 0
        assert self.lahns_score_power >= 0
        assert self.lahns_topk >= 0

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):


        state_dict.pop(prefix + "vfl_class_weights", None)
        state_dict.pop(prefix + "ccp_prototypes", None)
        state_dict.pop(prefix + "ccp_prototype_counts", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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
        negative_weight = self.alpha * pred_score.pow(self.gamma) * (1 - target)
        if self.use_class_aware_vfl and src_logits.shape[-1] == self.vfl_class_weights.numel():



            positive_class_weight = self.vfl_class_weights.to(
                device=src_logits.device, dtype=src_logits.dtype
            ).view(1, 1, -1)
            positive_weight = target_score * positive_class_weight
        else:

            positive_weight = target_score
        weight = negative_weight + positive_weight

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    def loss_labels_rank_gcl(self, outputs, targets, indices, num_boxes):
        """Rank-DETR GIoU-aware classification loss.

        Matched queries use normalized GIoU, ``(GIoU + 1) / 2``, as their
        soft classification target.  Unmatched queries and non-target class
        channels retain a zero target.  The focal factor ``|target-score|^gamma``
        teaches classification confidence to follow localization quality and
        suppresses high-confidence negatives.  Box quality is detached so this
        loss only optimizes the classification branch.
        """
        assert "pred_boxes" in outputs
        assert "pred_logits" in outputs

        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [target["boxes"][target_idx] for target, (_, target_idx) in zip(targets, indices)],
            dim=0,
        )

        if src_boxes.numel() > 0:
            matched_giou = torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes),
                    box_cxcywh_to_xyxy(target_boxes),
                )
            ).detach()
            matched_quality = ((matched_giou + 1.0) * 0.5).clamp_(0.0, 1.0)
        else:
            matched_quality = src_logits.new_zeros((0,))

        target_classes_o = torch.cat(
            [target["labels"][target_idx] for target, (_, target_idx) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = matched_quality.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = src_logits.sigmoid()
        focal_weight = (target_score - pred_score).abs().pow(self.rank_gcl_gamma)
        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, reduction="none"
        ) * focal_weight
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_rank_gcl": loss}

    def loss_labels_eqlv2_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert "pred_boxes" in outputs
        assert "pred_logits" in outputs

        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat(
                [target["boxes"][target_idx] for target, (_, target_idx) in zip(targets, indices)],
                dim=0,
            )
            if src_boxes.numel() > 0:
                ious = torch.diag(
                    box_iou(
                        box_cxcywh_to_xyxy(src_boxes),
                        box_cxcywh_to_xyxy(target_boxes),
                    )[0]
                ).detach()
            else:
                ious = src_logits.new_zeros((0,))
        else:
            ious = values

        target_classes_o = torch.cat(
            [target["labels"][target_idx] for target, (_, target_idx) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = src_logits.sigmoid().detach()
        vfl_weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        neg_weight = torch.sigmoid(
            self.eqlv2_gamma * (self.eqlv2_pos_neg - self.eqlv2_mu)
        ).to(device=src_logits.device, dtype=src_logits.dtype)
        pos_weight = 1.0 + self.eqlv2_alpha * (1.0 - neg_weight)
        eqlv2_weight = (
            pos_weight.view(1, 1, -1) * target
            + neg_weight.view(1, 1, -1) * (1 - target)
        )
        weight = vfl_weight * eqlv2_weight

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        with torch.no_grad():
            grad = (pred_score - target_score).abs() * weight
            pos_grad = (grad * target).sum(dim=(0, 1)).float()
            neg_grad = (grad * (1 - target)).sum(dim=(0, 1)).float()
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(pos_grad)
                torch.distributed.all_reduce(neg_grad)
            self.eqlv2_pos_grad.add_(pos_grad)
            self.eqlv2_neg_grad.add_(neg_grad)
            self.eqlv2_pos_neg.copy_(
                self.eqlv2_pos_grad / self.eqlv2_neg_grad.clamp_min(1e-10)
            )

        return {"loss_eqlv2_vfl": loss}

    def loss_labels_align(self, outputs, targets, indices, num_boxes):
        """Align-DETR IoU-aware classification loss (one-to-one ablation).

        The matched-class target combines the current class confidence ``p``
        and box IoU ``u`` as ``p**alpha * u**(1-alpha)``.  The target is
        detached, while unmatched class channels retain the focal negative
        weight ``p**gamma``.  This follows the official Align-DETR loss while
        deliberately leaving D-FINE matching and regression unchanged.
        """
        assert "pred_boxes" in outputs
        assert "pred_logits" in outputs

        src_logits = outputs["pred_logits"]
        pred_score = src_logits.sigmoid()
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [target["labels"][target_idx] for target, (_, target_idx) in zip(targets, indices)]
        )

        positive_weight = torch.zeros_like(src_logits)
        negative_weight = pred_score.pow(self.align_gamma)

        if target_classes_o.numel() > 0:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat(
                [
                    target["boxes"][target_idx]
                    for target, (_, target_idx) in zip(targets, indices)
                ],
                dim=0,
            )
            matched_iou = torch.diag(
                box_iou(
                    box_cxcywh_to_xyxy(src_boxes),
                    box_cxcywh_to_xyxy(target_boxes),
                )[0]
            ).clamp_(0.0, 1.0)

            positive_idx = (idx[0], idx[1], target_classes_o)
            matched_score = pred_score[positive_idx]
            quality_target = (
                matched_score.pow(self.align_quality_alpha)
                * matched_iou.pow(1.0 - self.align_quality_alpha)
            ).clamp_min_(self.align_min_quality).detach().to(dtype=src_logits.dtype)

            positive_weight[positive_idx] = quality_target
            negative_weight[positive_idx] = 1.0 - quality_target



        loss = -(
            positive_weight * F.logsigmoid(src_logits)
            + negative_weight * F.logsigmoid(-src_logits)
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_align": loss}

    def loss_class_confusion_prototype(self, outputs, targets, indices):
        """Separate matched query features with an EMA class-prototype bank.

        Incorrect classes already assigned a high classification probability
        are treated as harder negatives.  Only matched final-layer queries are
        used, so background queries and auxiliary decoder layers cannot
        dominate the objective.
        """
        query_features = outputs.get("query_features")
        if query_features is None:
            return None

        idx = self._get_src_permutation_idx(indices)
        features = query_features[idx]
        labels = torch.cat(
            [target["labels"][target_idx] for target, (_, target_idx) in zip(targets, indices)]
        )
        if features.numel() == 0:
            return query_features.sum() * 0.0
        if features.shape[-1] != self.ccp_feature_dim:
            raise ValueError(
                f"CCP expected feature dim {self.ccp_feature_dim}, got {features.shape[-1]}"
            )


        features = F.normalize(features.float(), dim=-1)
        with torch.no_grad():
            class_sums = features.detach().new_zeros(
                self.num_classes, self.ccp_feature_dim
            )
            class_counts = features.detach().new_zeros(self.num_classes)
            class_sums.index_add_(0, labels, features.detach())
            class_counts.index_add_(
                0, labels, torch.ones_like(labels, dtype=features.dtype)
            )
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(class_sums)
                torch.distributed.all_reduce(class_counts)

            present = class_counts > 0
            batch_prototypes = class_sums / class_counts.clamp_min(1).unsqueeze(-1)
            batch_prototypes = F.normalize(batch_prototypes, dim=-1)



            prototypes = self.ccp_prototypes.detach().clone().to(features)
            seen = self.ccp_prototype_counts > 0
            new_classes = present & ~seen
            prototypes[new_classes] = batch_prototypes[new_classes]
            valid_classes = seen | present

        prototype_logits = features @ F.normalize(prototypes, dim=-1).transpose(0, 1)
        prototype_logits = prototype_logits / self.ccp_temperature
        prototype_logits = prototype_logits.masked_fill(
            ~valid_classes.view(1, -1), torch.finfo(prototype_logits.dtype).min
        )

        if self.ccp_confusion_weight > 0 and "pred_logits" in outputs:
            confusion = outputs["pred_logits"][idx].detach().float().sigmoid()
            confusion.scatter_(1, labels.unsqueeze(1), 0.0)
            prototype_logits = (
                prototype_logits + self.ccp_confusion_weight * confusion
            )

        loss = F.cross_entropy(prototype_logits, labels)

        with torch.no_grad():
            old_present = present & seen
            updated = (
                self.ccp_momentum * self.ccp_prototypes[old_present]
                + (1.0 - self.ccp_momentum)
                * batch_prototypes[old_present].to(self.ccp_prototypes)
            )
            self.ccp_prototypes[old_present] = F.normalize(updated, dim=-1)
            self.ccp_prototypes[new_classes] = batch_prototypes[new_classes].to(
                self.ccp_prototypes
            )
            self.ccp_prototype_counts.add_(class_counts.to(self.ccp_prototype_counts))

        return loss * self.ccp_loss_weight

    def loss_orthogonal_projection(self, outputs, targets, indices):
        query_features = outputs.get("query_features")
        if query_features is None:
            return None

        idx = self._get_src_permutation_idx(indices)
        features = query_features[idx]
        labels = torch.cat(
            [target["labels"][target_idx] for target, (_, target_idx) in zip(targets, indices)]
        )
        if features.shape[0] < 2:
            return query_features.sum() * 0.0

        features = F.normalize(features.float(), p=2, dim=1)
        labels = labels.view(-1, 1)
        same_class = labels.eq(labels.transpose(0, 1))
        identity = torch.eye(
            same_class.shape[0], device=same_class.device, dtype=torch.bool
        )
        positive_mask = same_class.masked_fill(identity, False).to(features.dtype)
        negative_mask = (~same_class).to(features.dtype)
        similarities = features @ features.transpose(0, 1)

        positive_mean = (positive_mask * similarities).sum() / (
            positive_mask.sum() + 1e-6
        )
        negative_mean = (negative_mask * similarities).sum() / (
            negative_mask.sum() + 1e-6
        )
        loss = (1.0 - positive_mean) + self.opl_gamma * negative_mean
        return loss * self.opl_loss_weight

    def loss_localization_aware_hard_negatives(self, outputs, targets, indices):
        """Suppress confident unmatched queries in proportion to localization error.

        A query close to a real object may simply be a duplicate candidate, so
        it receives a small weight. A confident query with little overlap with
        any target receives the strongest penalty. Selection and weights are
        detached; gradients affect only the predicted top-class logit.
        """
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        total_loss = pred_logits.float().sum() * 0.0
        selected_count = 0

        for batch_index, ((matched_queries, _), target) in enumerate(
            zip(indices, targets)
        ):
            unmatched_mask = torch.ones(
                pred_logits.shape[1], dtype=torch.bool, device=pred_logits.device
            )
            unmatched_mask[matched_queries.to(unmatched_mask.device)] = False

            scores, classes = pred_logits[batch_index].detach().float().sigmoid().max(dim=-1)
            candidate_indices = torch.nonzero(
                unmatched_mask & (scores >= self.lahns_score_threshold),
                as_tuple=False,
            ).flatten()
            if candidate_indices.numel() == 0:
                continue

            candidate_boxes = pred_boxes[batch_index, candidate_indices].detach().float()
            target_boxes = target["boxes"].detach().float()
            if target_boxes.numel() > 0:
                ious, _ = box_iou(
                    box_cxcywh_to_xyxy(candidate_boxes),
                    box_cxcywh_to_xyxy(target_boxes),
                )
                max_iou = ious.max(dim=1).values.clamp(0, 1)
            else:
                max_iou = candidate_boxes.new_zeros(candidate_boxes.shape[0])

            candidate_scores = scores[candidate_indices]
            hardness = candidate_scores.pow(self.lahns_score_power) * (
                1.0 - max_iou
            ).pow(self.lahns_iou_power)

            if self.lahns_topk > 0 and candidate_indices.numel() > self.lahns_topk:
                _, keep = torch.topk(hardness, self.lahns_topk, sorted=False)
                candidate_indices = candidate_indices[keep]
                hardness = hardness[keep]

            candidate_classes = classes[candidate_indices]
            hard_logits = pred_logits[
                batch_index, candidate_indices, candidate_classes
            ].float()
            negative_loss = F.binary_cross_entropy_with_logits(
                hard_logits, torch.zeros_like(hard_logits), reduction="none"
            )
            total_loss = total_loss + (negative_loss * hardness).sum()
            selected_count += int(candidate_indices.numel())

        if selected_count == 0:
            return total_loss
        return total_loss * (self.lahns_loss_weight / selected_count)


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

        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
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

    def _get_num_boxes_from_indices(self, indices, device):
        num_boxes = sum(len(x[0]) for x in indices)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        return torch.clamp(num_boxes / get_world_size(), min=1).item()

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
            "eqlv2_vfl": self.loss_labels_eqlv2_vfl,
            "rank_gcl": self.loss_labels_rank_gcl,
            "align": self.loss_labels_align,
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

        if self.use_ccp_loss:
            loss_ccp = self.loss_class_confusion_prototype(outputs, targets, indices)
            if loss_ccp is not None:
                losses["loss_ccp"] = loss_ccp

        if self.use_opl:
            loss_opl = self.loss_orthogonal_projection(outputs, targets, indices)
            if loss_opl is not None:
                losses["loss_opl"] = loss_opl

        if self.use_lahns:
            losses["loss_lahns"] = self.loss_localization_aware_hard_negatives(
                outputs, targets, indices
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

                if self.use_o2m_aux:
                    indices_o2m = self.matcher(
                        aux_outputs,
                        targets,
                        return_topk=self.o2m_topk,
                        topk_metric=self.o2m_matcher_type,
                        quality_gamma=self.o2m_quality_gamma,
                        quality_min_pos=self.o2m_min_pos,
                    )["indices_o2m"]
                    num_boxes_o2m = self._get_num_boxes_from_indices(
                        indices_o2m, next(iter(outputs.values())).device
                    )
                    for loss in self.o2m_losses:
                        if loss == "local":
                            continue
                        meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_o2m)
                        l_dict = self.get_loss(
                            loss, aux_outputs, targets, indices_o2m, num_boxes_o2m, **meta
                        )
                        l_dict = {
                            k: l_dict[k] * self.weight_dict[k] * self.o2m_loss_weight
                            for k in l_dict
                            if k in self.weight_dict
                        }
                        l_dict = {k + f"_o2m_aux_{i}": v for k, v in l_dict.items()}
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
        elif loss in ("vfl", "eqlv2_vfl"):
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
