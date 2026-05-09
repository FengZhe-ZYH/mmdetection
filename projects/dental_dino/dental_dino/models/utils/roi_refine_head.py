# Copyright (c) OpenMMLab. RoI-based refinement head for SA-DINO v3.
"""Tooth-aware RoI refinement using high-resolution (stride-4) C2 features.

Design rationale:
- Dental lesions are small (AP_s dominant) with varying aspect ratios
  (median AR~1.12, max~3.1), so we use rectangular RoI Align (7x10)
  and ModulatedDeformConv to handle shape variation.
- Tooth mask provides anatomical context: we expand RoIs to cover the
  full tooth region rather than just the coarse detection box.
- A lightweight delta + IoU head refines localization in a cascade manner.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import ModulatedDeformConv2dPack
from torch import Tensor
from torchvision.ops import roi_align


class ToothRoIRefineHead(nn.Module):
    """Refine coarse detections using high-res C2 features + tooth context.

    Pipeline:  coarse_boxes -> tooth-expand -> RoI Align(C2) -> DCN -> delta+IoU

    Args:
        in_channels: C2 feature channels (before projection).
        feat_channels: Internal feature dim after projection.
        roi_h: RoI Align output height.
        roi_w: RoI Align output width.
        c2_stride: Backbone C2 stride w.r.t. input image.
        num_dcn_layers: Number of ModulatedDeformConv layers.
        num_tooth_classes: Number of tooth IDs for embedding.
        tooth_expand_ratio: How much to expand RoI toward tooth boundary.
        use_tooth_context: Whether to fuse tooth-ID embedding into RoI feats.
    """

    def __init__(
        self,
        in_channels: int = 256,
        feat_channels: int = 256,
        roi_h: int = 7,
        roi_w: int = 10,
        c2_stride: int = 4,
        num_dcn_layers: int = 2,
        num_tooth_classes: int = 34,
        tooth_expand_ratio: float = 0.3,
        use_tooth_context: bool = True,
    ) -> None:
        super().__init__()
        self.feat_channels = feat_channels
        self.roi_h = roi_h
        self.roi_w = roi_w
        self.c2_stride = c2_stride
        self.num_tooth_classes = num_tooth_classes
        self.tooth_expand_ratio = tooth_expand_ratio
        self.use_tooth_context = use_tooth_context

        self.c2_proj = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, feat_channels),
            nn.ReLU(inplace=True),
        )

        self.dcn_layers = nn.ModuleList()
        self.dcn_norms = nn.ModuleList()
        for _ in range(num_dcn_layers):
            self.dcn_layers.append(
                ModulatedDeformConv2dPack(
                    feat_channels, feat_channels, 3, padding=1))
            self.dcn_norms.append(nn.GroupNorm(32, feat_channels))

        if use_tooth_context:
            self.tooth_embed = nn.Embedding(num_tooth_classes, feat_channels)
            self.context_proj = nn.Linear(feat_channels, feat_channels)

        fc_in = feat_channels * roi_h * roi_w
        self.fc_delta = nn.Sequential(
            nn.Linear(fc_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
        )
        self.fc_iou = nn.Sequential(
            nn.Linear(fc_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.zeros_(self.fc_delta[-1].weight)
        nn.init.zeros_(self.fc_delta[-1].bias)
        nn.init.xavier_uniform_(self.fc_iou[-1].weight)
        nn.init.constant_(self.fc_iou[-1].bias, 0.0)

    def _expand_boxes_with_tooth(
        self,
        boxes_xyxy: Tensor,
        tooth_mask: Tensor,
        img_h: int,
        img_w: int,
    ) -> Tuple[Tensor, Tensor]:
        """Expand detection boxes to include tooth anatomical context.

        For each box, find the dominant tooth ID by majority vote in the box
        region, then expand the box toward the tooth's bounding box.

        Args:
            boxes_xyxy: [N, 4] absolute (x1, y1, x2, y2).
            tooth_mask: [B, 1, H, W] tooth ID mask (same size as input image).
            img_h, img_w: Input image spatial size.

        Returns:
            expanded_boxes: [N, 4] expanded (x1, y1, x2, y2).
            tooth_ids: [N] dominant tooth ID per box.
        """
        device = boxes_xyxy.device
        N = boxes_xyxy.shape[0]
        expanded = boxes_xyxy.clone()
        tooth_ids = torch.zeros(N, dtype=torch.long, device=device)

        mask_hw = tooth_mask[0, 0].long().clamp(0, self.num_tooth_classes - 1)

        for i in range(N):
            x1, y1, x2, y2 = boxes_xyxy[i].long()
            x1c = x1.clamp(0, img_w - 1)
            y1c = y1.clamp(0, img_h - 1)
            x2c = (x2 + 1).clamp(0, img_w)
            y2c = (y2 + 1).clamp(0, img_h)

            if x2c <= x1c or y2c <= y1c:
                continue

            region = mask_hw[y1c:y2c, x1c:x2c]
            valid = region[region > 0]
            if valid.numel() == 0:
                continue

            tid = valid.mode().values.item()
            tooth_ids[i] = tid

            tooth_region = (mask_hw == tid)
            tys, txs = torch.where(tooth_region)
            if len(tys) < 4:
                continue

            tx1 = txs.float().min()
            ty1 = tys.float().min()
            tx2 = txs.float().max()
            ty2 = tys.float().max()

            r = self.tooth_expand_ratio
            expanded[i, 0] = boxes_xyxy[i, 0] * (1 - r) + tx1 * r
            expanded[i, 1] = boxes_xyxy[i, 1] * (1 - r) + ty1 * r
            expanded[i, 2] = boxes_xyxy[i, 2] * (1 - r) + tx2 * r
            expanded[i, 3] = boxes_xyxy[i, 3] * (1 - r) + ty2 * r

        expanded[:, 0].clamp_(min=0)
        expanded[:, 1].clamp_(min=0)
        expanded[:, 2].clamp_(max=img_w)
        expanded[:, 3].clamp_(max=img_h)

        return expanded, tooth_ids

    def forward(
        self,
        c2_feat: Tensor,
        boxes_xyxy: Tensor,
        batch_indices: Tensor,
        tooth_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass: pool RoI features and predict delta + IoU.

        Args:
            c2_feat: [B, C, H/4, W/4] projected C2 features.
            boxes_xyxy: [N, 4] absolute (x1, y1, x2, y2) boxes.
            batch_indices: [N] batch index per box.
            tooth_ids: [N] tooth ID per box (optional, for context fusion).

        Returns:
            delta: [N, 4] predicted (dx, dy, dw, dh) offsets.
            iou_pred: [N, 1] predicted IoU score.
        """
        rois = torch.cat(
            [batch_indices.unsqueeze(1).to(boxes_xyxy.dtype), boxes_xyxy],
            dim=1)

        roi_feats = roi_align(
            c2_feat, rois,
            output_size=(self.roi_h, self.roi_w),
            spatial_scale=1.0 / self.c2_stride,
            sampling_ratio=2)

        for dcn, norm in zip(self.dcn_layers, self.dcn_norms):
            roi_feats = F.relu(norm(dcn(roi_feats)), inplace=True)

        if self.use_tooth_context and tooth_ids is not None:
            ids_clamped = tooth_ids.clamp(0, self.num_tooth_classes - 1)
            t_emb = self.tooth_embed(ids_clamped)
            t_emb = self.context_proj(t_emb)
            roi_feats = roi_feats + t_emb[:, :, None, None]

        x = roi_feats.flatten(1)
        delta = self.fc_delta(x)
        iou_pred = self.fc_iou(x)

        return delta, iou_pred


def apply_delta_to_boxes(
    boxes_cxcywh: Tensor,
    delta: Tensor,
    max_delta: float = 2.0,
) -> Tensor:
    """Apply predicted deltas to boxes in (cx, cy, w, h) format.

    delta encoding: (dx, dy, dw, dh) where
        cx' = cx + dx * w,  cy' = cy + dy * h
        w'  = w * exp(dw),  h'  = h * exp(dh)
    """
    delta = delta.clamp(-max_delta, max_delta)
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    dx, dy, dw, dh = delta.unbind(-1)

    new_cx = cx + dx * w
    new_cy = cy + dy * h
    new_w = w * dw.exp()
    new_h = h * dh.exp()

    return torch.stack([new_cx, new_cy, new_w, new_h], dim=-1)


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], -1)


def refine_loss(
    delta: Tensor,
    iou_pred: Tensor,
    pred_boxes_cxcywh: Tensor,
    gt_boxes_cxcywh: Tensor,
    loss_delta_weight: float = 5.0,
    loss_giou_weight: float = 2.0,
    loss_iou_weight: float = 1.0,
) -> dict:
    """Compute refinement losses for matched positive pairs.

    Returns dict with loss_refine_delta, loss_refine_giou, loss_refine_iou.
    """
    if pred_boxes_cxcywh.numel() == 0:
        zero = pred_boxes_cxcywh.sum() * 0.0
        return dict(
            loss_refine_delta=zero,
            loss_refine_giou=zero,
            loss_refine_iou=zero)

    refined_cxcywh = apply_delta_to_boxes(pred_boxes_cxcywh, delta)
    refined_xyxy = cxcywh_to_xyxy(refined_cxcywh)
    gt_xyxy = cxcywh_to_xyxy(gt_boxes_cxcywh)

    loss_l1 = F.l1_loss(refined_cxcywh, gt_boxes_cxcywh, reduction='mean')

    loss_giou = _giou_loss(refined_xyxy, gt_xyxy)

    with torch.no_grad():
        actual_iou = _box_iou(refined_xyxy.detach(), gt_xyxy)
    loss_iou = F.binary_cross_entropy_with_logits(
        iou_pred.squeeze(-1), actual_iou, reduction='mean')

    return dict(
        loss_refine_delta=loss_delta_weight * loss_l1,
        loss_refine_giou=loss_giou_weight * loss_giou,
        loss_refine_iou=loss_iou_weight * loss_iou,
    )


def _box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute element-wise IoU between two sets of boxes in xyxy format."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    return inter / (area1 + area2 - inter).clamp(min=1e-6)


def _giou_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Compute mean GIoU loss."""
    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_t = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])

    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = (area_p + area_t - inter).clamp(min=1e-6)

    enclosing_lt = torch.min(pred[:, :2], target[:, :2])
    enclosing_rb = torch.max(pred[:, 2:], target[:, 2:])
    enclosing_wh = (enclosing_rb - enclosing_lt).clamp(min=0)
    enclosing_area = (enclosing_wh[:, 0] * enclosing_wh[:, 1]).clamp(min=1e-6)

    giou = inter / union - (enclosing_area - union) / enclosing_area
    return (1 - giou).mean()
