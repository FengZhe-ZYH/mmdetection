# Copyright (c) OpenMMLab. Dental auxiliary losses: heatmap + NWD-style box loss.

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor


def gaussian_heatmap_target(
        batch_instances: List[InstanceData],
        batch_img_metas: List[dict],
        heatmap: Tensor,
        eps: float = 1e-6,
) -> Tensor:
    """Place Gaussian peaks at bbox centers (xyxy pixel / img_shape -> grid)."""
    b, _, hm, wm = heatmap.shape
    if len(batch_instances) != b or len(batch_img_metas) != b:
        raise ValueError('batch size mismatch')
    device = heatmap.device
    dtype = heatmap.dtype
    tgt = torch.zeros((b, 1, hm, wm), device=device, dtype=dtype)
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, hm, device=device, dtype=dtype),
        torch.linspace(0, 1, wm, device=device, dtype=dtype),
        indexing='ij')
    grid = torch.stack([xx, yy], dim=-1)

    for i, inst in enumerate(batch_instances):
        h_img = float(batch_img_metas[i]['img_shape'][0])
        w_img = float(batch_img_metas[i]['img_shape'][1])
        if h_img <= eps or w_img <= eps:
            raise RuntimeError('invalid img_shape for heatmap target')
        if not hasattr(inst, 'bboxes') or inst.bboxes is None:
            continue
        boxes = inst.bboxes
        if hasattr(boxes, 'tensor'):
            boxes = boxes.tensor
        if boxes.numel() == 0:
            continue
        for bi in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[bi]
            cx = ((x1 + x2) / 2.0) / w_img
            cy = ((y1 + y2) / 2.0) / h_img
            sigma = 0.04
            d2 = (grid[..., 0] - cx)**2 + (grid[..., 1] - cy)**2
            peak = torch.exp(-d2 / (2 * sigma * sigma))
            tgt[i, 0] = torch.maximum(tgt[i, 0], peak)
    return tgt.clamp(0, 1)


def heatmap_focal_loss(pred: Tensor, target: Tensor, alpha: float = 0.25,
                       gamma: float = 2.0) -> Tensor:
    """Simple focal-style loss on heatmap logits (pred before sigmoid)."""
    pred_sig = pred.sigmoid()
    ce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p_t = pred_sig * target + (1 - pred_sig) * (1 - target)
    loss = ce * ((1 - p_t)**gamma)
    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss
    return loss.mean()


def pairwise_w2_diag_gaussians(cxcywh_a: Tensor, cxcywh_b: Tensor) -> Tensor:
    """W2^2 between sets of boxes; cxcywh_a [Q,4], cxcywh_b [N,4] -> [Q,N]."""
    qa, _ = cxcywh_a.shape
    nb, _ = cxcywh_b.shape
    a = cxcywh_a.view(qa, 1, 4)
    b = cxcywh_b.view(1, nb, 4)
    cx1, cy1, w1, h1 = a.unbind(-1)
    cx2, cy2, w2, h2 = b.unbind(-1)
    s1x = (w1.clamp(min=1e-4) / 4.0)**2
    s1y = (h1.clamp(min=1e-4) / 4.0)**2
    s2x = (w2.clamp(min=1e-4) / 4.0)**2
    s2y = (h2.clamp(min=1e-4) / 4.0)**2
    mean_term = (cx1 - cx2)**2 + (cy1 - cy2)**2
    var_term = s1x + s2x - 2 * (s1x * s2x).sqrt() + s1y + s2y - 2 * (s1y *
                                                                      s2y).sqrt()
    return mean_term + var_term


def nwd_aux_loss(
        pred_boxes: Tensor,
        batch_gt_instances: List[InstanceData],
        batch_img_metas: List[dict],
        *,
        small_area_thr: float = 0.001,
        eps: float = 1e-6,
) -> Tensor:
    """For small GT boxes (normalized area < thr), min over queries of W2^2."""
    if pred_boxes.dim() != 3 or pred_boxes.size(-1) != 4:
        raise ValueError(f'pred_boxes [B,Q,4] expected, got {tuple(pred_boxes.shape)}')
    b, _, _ = pred_boxes.shape
    terms: List[Tensor] = []
    from mmdet.structures.bbox import bbox_xyxy_to_cxcywh

    for i in range(b):
        inst = batch_gt_instances[i]
        h_img = float(batch_img_metas[i]['img_shape'][0])
        w_img = float(batch_img_metas[i]['img_shape'][1])
        area_img = h_img * w_img
        if area_img < eps:
            raise RuntimeError('invalid img_shape')
        if not hasattr(inst, 'bboxes') or inst.bboxes is None:
            continue
        boxes = inst.bboxes
        if hasattr(boxes, 'tensor'):
            boxes = boxes.tensor
        if boxes.numel() == 0:
            continue
        gt_cxcywh = bbox_xyxy_to_cxcywh(boxes) / boxes.new_tensor(
            [w_img, h_img, w_img, h_img])
        for j in range(gt_cxcywh.shape[0]):
            gw, gh = gt_cxcywh[j, 2], gt_cxcywh[j, 3]
            if (gw * gh) < small_area_thr:
                g = gt_cxcywh[j:j + 1]
                d2 = pairwise_w2_diag_gaussians(pred_boxes[i], g)
                terms.append(d2.min())

    if not terms:
        return pred_boxes.sum() * 0.0
    return torch.stack(terms).mean()


class HeatmapHead(nn.Module):

    def __init__(self, in_channels: int = 256) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)
