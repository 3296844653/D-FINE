"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ...core import register
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou


@register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = [
        "use_focal_loss",
    ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict["cost_class"]
        self.cost_bbox = weight_dict["cost_bbox"]
        self.cost_giou = weight_dict["cost_giou"]

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert (
            self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets,
        return_topk=False,
        topk_metric="cost",
        quality_gamma=0.25,
        quality_min_pos=1,
    ):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = (
                outputs["pred_logits"].flatten(0, 1).softmax(-1)
            )  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (
                (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            )
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix 3 * self.cost_bbox + 2 * self.cost_class + self.cost_giou
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        C = torch.nan_to_num(C, nan=1.0)
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices_pre
        ]

        # Compute topk indices
        if return_topk:
            return {
                "indices_o2m": self.get_top_k_matches(
                    C, sizes=sizes, k=return_topk, initial_indices=indices_pre
                )
                if topk_metric == "cost"
                else self.get_quality_top_k_matches(
                    outputs,
                    targets,
                    sizes=sizes,
                    k=return_topk,
                    gamma=quality_gamma,
                    min_pos=quality_min_pos,
                )
            }

        return {"indices": indices}  # , 'indices_o2m': C.min(-1)[1]}

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        cost_per_image = [c[i].clone() for i, c in enumerate(C.split(sizes, -1))]
        matches_per_image = [[] for _ in sizes]

        for round_idx in range(k):
            for image_idx, cost in enumerate(cost_per_image):
                if cost.numel() == 0:
                    row_ind, col_ind = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
                elif round_idx == 0 and initial_indices is not None:
                    row_ind, col_ind = initial_indices[image_idx]
                else:
                    row_ind, col_ind = linear_sum_assignment(cost.numpy())

                row_ind = torch.as_tensor(row_ind, dtype=torch.int64)
                col_ind = torch.as_tensor(col_ind, dtype=torch.int64)
                matches_per_image[image_idx].append((row_ind, col_ind))

                if row_ind.numel() > 0:
                    # O2M should add more positive queries per target while avoiding
                    # assigning the exact same prediction again in later rounds.
                    cost[row_ind, :] = 1e6

        return [
            (
                torch.cat([match[0] for match in image_matches], dim=0),
                torch.cat([match[1] for match in image_matches], dim=0),
            )
            for image_matches in matches_per_image
        ]

    def get_quality_top_k_matches(self, outputs, targets, sizes, k=1, gamma=0.25, min_pos=1):
        """PaQ-DETR style quality-aware O2M assignment."""

        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        bs, num_queries = pred_logits.shape[:2]

        if self.use_focal_loss:
            pred_scores = pred_logits.sigmoid()
        else:
            pred_scores = pred_logits.softmax(-1)

        candidate_topk = max(int(k), 1)
        min_pos = max(int(min_pos), 1)
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        matches_per_image = []

        for image_idx in range(bs):
            num_targets = sizes[image_idx]
            if num_targets == 0:
                empty = torch.empty(0, dtype=torch.int64)
                matches_per_image.append((empty, empty))
                continue

            target_labels = targets[image_idx]["labels"]
            target_boxes_xyxy = box_cxcywh_to_xyxy(targets[image_idx]["boxes"])

            cls_quality = pred_scores[image_idx][:, target_labels]
            loc_quality = box_iou(pred_boxes_xyxy[image_idx], target_boxes_xyxy)[0].clamp(min=0)
            quality = (loc_quality - gamma * cls_quality).detach().cpu()

            topk = min(candidate_topk, num_queries)
            quality_topk = torch.topk(quality, k=topk, dim=0).values
            target_limits = torch.ceil(quality_topk.sum(dim=0)).to(torch.long)
            target_limits = target_limits.clamp(min=min_pos, max=topk)

            selected_rows, selected_cols = [], []
            target_counts = torch.zeros(num_targets, dtype=torch.long)
            used_queries = torch.zeros(num_queries, dtype=torch.bool)
            order = torch.argsort(quality.flatten(), descending=True)

            for flat_idx in order:
                row = torch.div(flat_idx, num_targets, rounding_mode="floor").item()
                col = (flat_idx % num_targets).item()
                if bool(used_queries[row].item()):
                    continue
                if target_counts[col].item() >= target_limits[col].item():
                    continue
                selected_rows.append(row)
                selected_cols.append(col)
                used_queries[row] = True
                target_counts[col] += 1
                if bool(torch.all(target_counts >= target_limits).item()):
                    break

            matches_per_image.append(
                (
                    torch.as_tensor(selected_rows, dtype=torch.int64),
                    torch.as_tensor(selected_cols, dtype=torch.int64),
                )
            )

        return matches_per_image
